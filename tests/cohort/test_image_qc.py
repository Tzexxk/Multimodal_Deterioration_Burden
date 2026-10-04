"""Synthetic DICOM transforms, QC review and recoverable Step 07 publication."""

from __future__ import annotations

import copy
import csv
import io
import json
import os
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest.mock import PropertyMock, patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import test_build_labels as fixtures
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import (
    ComputedRadiographyImageStorage,
    DigitalXRayImageStorageForPresentation,
    DigitalXRayImageStorageForProcessing,
    ExplicitVRLittleEndian,
    RLELossless,
    generate_uid,
)

from src.cohort.common import _file_hash, _json_file, _OutputRun, _verify_run
from src.imaging.prepare_images import (
    REVIEW_COLUMNS,
    assign_qc,
    load_inputs,
    prepare_images,
)
from src.imaging.preprocessing import (
    process_dataset,
    validate_pixels,
    validate_settings,
)

PROJECT = Path(__file__).parents[2]


def dicom(values=None, photo="MONOCHROME2", *, frames=1, color=False):
    if values is None:
        values = np.arange(620, dtype=np.uint16).reshape(20, 31)
    values = np.asarray(values)
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = ComputedRadiographyImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset("", {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Rows = values.shape[-3] if color else values.shape[-2]
    ds.Columns = values.shape[-2] if color else values.shape[-1]
    ds.SamplesPerPixel = 3 if color else 1
    ds.NumberOfFrames = frames
    ds.PhotometricInterpretation = photo
    if color:
        ds.PlanarConfiguration = 0
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = int(values.dtype.kind == "i")
    ds.PixelData = values.astype("<i2" if ds.PixelRepresentation else "<u2").tobytes()
    return ds


def lut(values, first=0, bits=16):
    item = Dataset()
    item.LUTDescriptor = [len(values), first, bits]
    item.add_new(0x00283006, "US", list(map(int, values)))
    return item


class ImageTransformTests(unittest.TestCase):
    def setUp(self):
        self.settings = _json_file(PROJECT / "configs/image_processing_v1.json")

    def test_direction_presentation_and_odd_padding(self):
        a = process_dataset(dicom(), self.settings)
        b = process_dataset(dicom(photo="MONOCHROME1"), self.settings)
        ds = dicom(photo="MONOCHROME1")
        ds.PresentationLUTShape = "INVERSE"
        c = process_dataset(ds, self.settings)
        self.assertEqual(c["processing_status"], "complete")
        np.testing.assert_array_equal(b["pixels"], c["pixels"])
        self.assertEqual(c["presentation_path"], "shape_inverse")
        self.assertTrue(c["inversion_applied"])
        self.assertEqual(a["resized_height"], 145)
        self.assertEqual(a["pad_top"], 39)
        self.assertEqual(a["pad_bottom"], 40)
        np.testing.assert_allclose(
            a["pixels"][39:184] + b["pixels"][39:184], 1, atol=2e-7
        )
        self.assertTrue((a["pixels"][:39] == 0).all())
        ds.PresentationLUTShape = "IDENTITY"
        conflict = process_dataset(ds, self.settings)
        self.assertEqual(conflict["processing_status"], "pending_review")
        self.assertIsNone(conflict["pixels"])
        self.assertIsNotNone(conflict["before"])
        ds = dicom()
        ds.PresentationLUTSequence = [lut(range(620))]
        presented = process_dataset(ds, self.settings)
        self.assertEqual(presented["presentation_path"], "presentation_lut")
        ds.PhotometricInterpretation = "MONOCHROME1"
        np.testing.assert_array_equal(
            process_dataset(ds, self.settings)["pixels"], presented["pixels"]
        )
        ds.PresentationLUTShape = "INVERSE"
        self.assertEqual(
            process_dataset(ds, self.settings)["error_code"], "invalid_presentation_lut"
        )

    def test_lut_priority_valid_fallbacks_and_rescale_defaults(self):
        ds = dicom()
        ds.RescaleSlope = 3
        ds.RescaleIntercept = 10
        ds.ModalityLUTSequence = [lut(range(620))]
        ds.VOILUTSequence = [lut(range(620))]
        ds.WindowCenter = 300
        ds.WindowWidth = 200
        r = process_dataset(ds, self.settings)
        self.assertEqual(r["processing_status"], "complete")
        self.assertEqual(r["modality_path"], "modality_lut")
        self.assertEqual(r["voi_path"], "voi_lut")
        ds.VOILUTSequence = [Dataset(), lut(range(620))]
        r = process_dataset(ds, self.settings)
        self.assertEqual(r["display_metadata"]["voi_lut_index"], 1)
        self.assertIn("invalid_voi_lut_skipped", r["issues"])
        ds.VOILUTSequence = [Dataset()]
        r = process_dataset(ds, self.settings)
        self.assertEqual(r["voi_path"], "window")
        self.assertIn("invalid_voi_lut_fallback", r["issues"])
        ds = dicom()
        ds.RescaleSlope = 2
        r = process_dataset(ds, self.settings)
        self.assertEqual(r["modality_path"], "rescale")
        self.assertEqual(r["display_metadata"]["applied_rescale_intercept"], 0)
        self.assertEqual(r["voi_path"], "none")
        ds.WindowCenter = [200, 500]
        ds.WindowWidth = [100, 500]
        r = process_dataset(ds, self.settings)
        self.assertEqual(r["display_metadata"]["applied_window_center"], 200)
        ds.WindowWidth = [0, 500]
        r = process_dataset(ds, self.settings)
        self.assertEqual(r["display_metadata"]["window_index"], 1)
        self.assertEqual(r["display_metadata"]["applied_window_center"], 500)
        self.assertIn("invalid_window_group_skipped", r["issues"])

    def test_padding_range_percentiles_and_mask(self):
        values = np.arange(-31, 589, dtype=np.int16).reshape(20, 31)
        ds = dicom(values)
        ds.PixelPaddingValue = -31
        ds.PixelPaddingRangeLimit = -1
        r = process_dataset(ds, self.settings)
        expected = np.percentile(values[values >= 0], [1, 99], method="linear")
        np.testing.assert_allclose([r["p1"], r["p99"]], expected)
        self.assertEqual(r["valid_pixel_fraction"], 19 / 20)
        self.assertTrue((r["pixels"][r["pad_top"] : r["pad_top"] + 3] == 0).all())
        ds.PixelPaddingValue = -1
        ds.PixelPaddingRangeLimit = -31
        np.testing.assert_array_equal(
            process_dataset(ds, self.settings)["pixels"], r["pixels"]
        )
        ds = dicom(np.full((20, 31), 10, dtype=np.uint16))
        ds.PixelPaddingValue = 10
        self.assertEqual(
            process_dataset(ds, self.settings)["error_code"], "no_valid_pixels"
        )

    def test_dx_voi_depth_specialization_and_priority(self):
        for sop in (
            DigitalXRayImageStorageForPresentation,
            DigitalXRayImageStorageForProcessing,
        ):
            for bits in range(10, 17):
                with self.subTest(sop=sop, bits=bits):
                    ds = dicom()
                    ds.SOPClassUID = sop
                    ds.VOILUTSequence = [lut(np.arange(620) ** 2 // 619, bits=bits)]
                    ds.WindowCenter, ds.WindowWidth = 300, 40
                    r = process_dataset(ds, self.settings)
                    self.assertEqual(r["processing_status"], "complete")
                    self.assertEqual(r["voi_path"], "voi_lut")
                    self.assertEqual(r["issues"], [])
                    self.assertEqual(
                        r["display_metadata"]["voi_lut_descriptor"][2], bits
                    )
                    ds.WindowWidth = 500
                    np.testing.assert_array_equal(
                        process_dataset(ds, self.settings)["pixels"], r["pixels"]
                    )
        ds = dicom()
        ds.SOPClassUID = DigitalXRayImageStorageForPresentation
        ds.VOILUTSequence = [lut(range(620), bits=9)]
        ds.WindowCenter, ds.WindowWidth = 300, 500
        r = process_dataset(ds, self.settings)
        self.assertEqual(r["voi_path"], "window")
        self.assertIn("invalid_voi_lut_fallback", r["issues"])
        ds.VOILUTSequence = [lut([0] * 619 + [2**14], bits=14)]
        self.assertIn(
            "invalid_voi_lut_fallback", process_dataset(ds, self.settings)["issues"]
        )
        ds.SOPClassUID = ComputedRadiographyImageStorage
        ds.VOILUTSequence = [lut(range(620), bits=14)]
        self.assertIn(
            "invalid_voi_lut_fallback", process_dataset(ds, self.settings)["issues"]
        )

    def test_failures_never_return_substitute_pixels(self):
        cases = [
            dicom(np.ones((20, 31), dtype=np.uint16)),
            dicom(np.arange(1240, dtype=np.uint16).reshape(2, 20, 31), frames=2),
            dicom(
                np.arange(1860, dtype=np.uint16).reshape(20, 31, 3),
                photo="RGB",
                color=True,
            ),
        ]
        missing = dicom()
        del missing.PixelData
        cases.append(missing)
        for ds in cases:
            r = process_dataset(ds, self.settings)
            self.assertEqual(r["processing_status"], "fail")
            self.assertIsNone(r["pixels"])
        with patch(
            "pydicom.dataset.Dataset.pixel_array",
            new_callable=PropertyMock,
            return_value=np.full((20, 31), np.nan),
        ):
            r = process_dataset(dicom(), self.settings)
            self.assertEqual(r["error_code"], "nonfinite_pixels")
        with patch(
            "pydicom.dataset.Dataset.pixel_array",
            new_callable=PropertyMock,
            side_effect=ImportError("decoder missing"),
        ):
            self.assertIn(
                "pixel_decode_error",
                process_dataset(dicom(), self.settings)["error_code"],
            )

    def test_compressed_decode_and_numeric_contract(self):
        ds = dicom()
        expected = process_dataset(ds, self.settings)["pixels"]
        ds.compress(RLELossless)
        result = process_dataset(ds, self.settings)
        self.assertEqual(result["processing_status"], "complete")
        np.testing.assert_array_equal(expected, result["pixels"])
        validate_pixels(result["pixels"])
        with self.assertRaises(ValueError):
            validate_pixels(np.zeros((224, 224), dtype=np.float64))
        settings = {**self.settings, "percentiles": [0, 100]}
        with self.assertRaises(ValueError):
            validate_settings(settings)

    def test_sampling_exact_duplicates_and_hash_bound_decisions(self):
        records = []
        for number in range(12):
            records.append(
                dict(
                    sample_id=f"s{number}",
                    view_position="AP",
                    photometric_interpretation="MONOCHROME2",
                    modality_path="identity",
                    voi_path="none",
                    presentation_path="photometric",
                    issues=[],
                    error_code="",
                    processing_status="complete",
                    source_dicom_sha256=f"source{number}",
                    decoded_pixel_sha256=f"pixel{number}",
                    npy_sha256=f"npy{number}",
                    processing_config_sha256="config",
                )
            )
        original = copy.deepcopy(records)
        assign_qc(records, self.settings, {})
        selected = {r["sample_id"] for r in records if r["sampled_for_review"]}
        self.assertEqual(len(selected), 5)
        self.assertEqual(sum(r["qc_status"] == "pass" for r in records), 7)
        reversed_records = list(reversed(copy.deepcopy(original)))
        assign_qc(reversed_records, self.settings, {})
        self.assertEqual(
            selected,
            {r["sample_id"] for r in reversed_records if r["sampled_for_review"]},
        )
        row = records[0]
        decision = {k: row[k] for k in REVIEW_COLUMNS[:4]}
        decision.update(
            decision="pass",
            rationale="Synthetic review",
            reviewer="Synthetic",
            reviewed_on="2026-10-04",
            version="v1",
        )
        records[1]["decoded_pixel_sha256"] = records[0]["decoded_pixel_sha256"]
        duplicates = assign_qc(records, self.settings, {row["sample_id"]: decision})
        self.assertEqual(len(duplicates), 2)
        self.assertEqual(row["qc_status"], "pass")
        self.assertEqual(records[1]["qc_status"], "pending_review")
        decision["npy_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "Stale"):
            assign_qc(records, self.settings, {row["sample_id"]: decision})


class ImagePublicationTests(unittest.TestCase):
    def setUp(self):
        self.labels_fixture = fixtures.FormalLabelTests()
        self.labels_fixture.setUp()
        self.addCleanup(self.labels_fixture.doCleanups)
        self.f = self.labels_fixture.fixture
        self.root = self.f.root
        self.configs = self.labels_fixture.configs
        self.processing = self.root / "configs/image_processing_v1.json"
        self.processing.write_bytes(
            (PROJECT / "configs/image_processing_v1.json").read_bytes()
        )
        self.qc = self.f.outputs / "qc/burden_v2"
        self.image_dir = self.f.processed / "imaging/burden_v2"
        self.sources = {}

    def prepare(self, duplicate=False):
        self.f.prepare()
        index = []
        selected = []
        old = pq.read_table(
            self.f.processed / "index/study_image_selection.parquet"
        ).to_pylist()
        for i, row in enumerate(old):
            relative = f"dataset/p{row['subject_id']}/s{row['study_id']}/{row['selected_dicom_id']}.dcm"
            row["dicom_path"] = relative
            match = next(m for m in self.f.matches if m["study_id"] == row["study_id"])
            time = datetime.fromisoformat(str(match["t0"]))
            seed = 100 if duplicate and i < 2 else row["study_id"]
            values = np.random.default_rng(seed).integers(
                0, 2000, (20, 31), dtype=np.uint16
            )
            ds = dicom(values)
            ds.ViewPosition = "AP"
            ds.AcquisitionDateTime = time.strftime("%Y%m%d%H%M%S")
            path = self.f.cxr / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            ds.save_as(path, enforce_file_format=True)
            self.sources[row["study_id"]] = path
            selected.append(row)
            index.append(
                dict(
                    subject_id=row["subject_id"],
                    study_id=row["study_id"],
                    dicom_id=row["selected_dicom_id"],
                    dicom_path=relative,
                    view_position="AP",
                    acquisition_datetime=match["t0"],
                    rows=20,
                    columns=31,
                )
            )
        index_path = self.f.processed / "index/cxr_index.parquet"
        with _OutputRun(self.f.outputs, 2, self.f.processed) as run:
            pq.write_table(pa.Table.from_pylist(index), run.path(index_path))
        selection = self.f.processed / "index/study_image_selection.parquet"
        with _OutputRun(self.f.outputs, 3, self.f.processed) as run:
            pq.write_table(pa.Table.from_pylist(selected), run.path(selection))
            run.path(
                self.f.outputs / "audit/study_image_selection_summary.json"
            ).write_text(
                json.dumps(dict(input_parquet_sha256=_file_hash(index_path))),
                encoding="utf-8",
            )
        summary = self.f.outputs / "audit/admission_match_summary.csv"
        with summary.open(encoding="utf-8", newline="") as stream:
            items = list(csv.DictReader(stream))
        for r in items:
            if r["metric"] == "selection_input_sha256":
                r["value"] = _file_hash(selection)
        manifest_path = self.f.outputs / "audit/step_04_manifest.json"
        with summary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["metric", "value"])
            writer.writeheader()
            writer.writerows(items)
        manifest = _json_file(manifest_path)
        manifest["files"][str(summary)] = _file_hash(summary)
        self.labels_fixture.write_json(manifest_path, manifest)
        self.f.run_audit()
        self.labels_fixture.freeze()
        self.labels_fixture.run_labels()

    def run_images(self, reviews=None):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(
                prepare_images(
                    self.root,
                    self.configs[0],
                    self.configs[1],
                    self.configs[2],
                    self.configs[3],
                    self.processing,
                    reviews,
                ),
                0,
            )
        _verify_run(self.f.outputs, 7)
        return pq.read_table(self.image_dir / "image_qc_status.parquet").to_pylist()

    def review_file(self, decisions):
        path = self.root / "configs/reviews/image_qc_v1.csv"
        path.parent.mkdir(exist_ok=True)
        with (self.qc / "image_qc_review.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            template = list(csv.DictReader(stream))
        rows = []
        for row in template:
            row = {k: row[k] for k in REVIEW_COLUMNS}
            row.update(
                decision=decisions.get(row["sample_id"], "pass"),
                rationale="Synthetic image review",
                reviewer="Synthetic",
                reviewed_on="2026-10-04",
                version="image_qc_v1",
            )
            rows.append(row)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_publication_review_repeat_and_rollback(self):
        self.prepare()
        frozen = {
            str(self.f.outputs / f"audit/step_{s:02d}_manifest.json"): _file_hash(
                self.f.outputs / f"audit/step_{s:02d}_manifest.json"
            )
            for s in range(2, 7)
        }
        frozen[str(self.labels_fixture.audit / "endpoint_decision.json")] = _file_hash(
            self.labels_fixture.audit / "endpoint_decision.json"
        )
        records = self.run_images()
        self.assertEqual(len(records), 3)
        self.assertTrue(all(r["qc_status"] == "pending_review" for r in records))
        schema = pq.read_schema(self.image_dir / "image_processing.parquet")
        self.assertEqual(schema.field("t0").type, pa.timestamp("us"))
        for r in records:
            validate_pixels(
                np.load(self.image_dir / r["npy_relpath"], allow_pickle=False)
            )
        html = (self.qc / "image_qc_review.html").read_text(encoding="utf-8")
        self.assertNotIn("burden_score", html)
        self.assertNotIn("phys_death", html)
        self.assertEqual(html.count("<article "), 3)
        repeated = self.run_images()
        self.assertEqual(
            [r["npy_sha256"] for r in records], [r["npy_sha256"] for r in repeated]
        )
        reviews = self.review_file({})
        accepted = self.run_images(reviews)
        self.assertTrue(
            all(r["qc_status"] == "pass" and r["review_complete"] for r in accepted)
        )
        self.assertTrue(
            _json_file(self.qc / "run_metadata.json")["ready_for_cohort_freeze"]
        )
        old = _json_file(self.f.outputs / "audit/step_07_manifest.json")
        original = os.replace

        def fail_one(source, target):
            if (
                str(target).endswith("image_qc_status.parquet")
                and ".step_07_stage_" in Path(source).name
            ):
                raise OSError("synthetic image publication failure")
            return original(source, target)

        with patch("src.cohort.common.os.replace", side_effect=fail_one):
            with self.assertRaises(OSError):
                self.run_images(reviews)
        self.assertEqual(
            old, _json_file(self.f.outputs / "audit/step_07_manifest.json")
        )
        _verify_run(self.f.outputs, 7)
        for p, h in frozen.items():
            self.assertEqual(_file_hash(Path(p)), h)
        self.assertFalse((self.image_dir / "image_manifest.parquet").exists())

    def test_missing_source_stale_cache_and_review_restrictions(self):
        self.prepare()
        previous = self.run_images()
        missing = next(r for r in previous if r["study_id"] == 200)
        self.sources[200].unlink()
        rows = self.run_images()
        failed = next(r for r in rows if r["study_id"] == 200)
        self.assertEqual(failed["qc_status"], "fail")
        self.assertIsNone(failed["npy_relpath"])
        self.assertTrue((self.image_dir / missing["npy_relpath"]).is_file())
        files = _json_file(self.f.outputs / "audit/step_07_manifest.json")["files"]
        self.assertNotIn(str(self.image_dir / missing["npy_relpath"]), files)
        reviews = self.review_file({})
        with self.assertRaisesRegex(ValueError, "Manual pass"):
            self.run_images(reviews)
        reviews = self.review_file({failed["sample_id"]: "exclude"})
        accepted = self.run_images(reviews)
        self.assertTrue(all(r["review_complete"] for r in accepted))
        self.assertEqual(sum(r["qc_status"] == "pass" for r in accepted), 2)

    def test_duplicates_and_stale_review_hash(self):
        self.prepare(duplicate=True)
        rows = self.run_images()
        self.assertEqual(
            _json_file(self.qc / "run_metadata.json")["duplicate_studies"], 2
        )
        self.assertTrue(
            all(
                "duplicate_image" in r["qc_reason"]
                for r in rows
                if r["study_id"] in (100, 101)
            )
        )
        reviews = self.review_file({})
        self.run_images(reviews)
        reviews.write_text(
            reviews.read_text(encoding="utf-8").replace(
                rows[0]["npy_sha256"], "f" * 64
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Stale"):
            self.run_images(reviews)

    def test_changed_header_config_and_review_schema_preserve_outputs(self):
        self.prepare()
        self.run_images()
        old = _json_file(self.f.outputs / "audit/step_07_manifest.json")
        path = self.sources[100]
        original = path.read_bytes()
        import pydicom

        ds = pydicom.dcmread(path)
        ds.ViewPosition = "PA"
        ds.save_as(path, enforce_file_format=True)
        with self.assertRaisesRegex(ValueError, "metadata changed"):
            self.run_images()
        path.write_bytes(original)
        reviews = self.root / "configs/bad_reviews.csv"
        reviews.write_text("sample_id,decision\ns1,pass\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "columns"):
            self.run_images(reviews)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.run_images(self.qc / "image_qc_review.csv")
        config = _json_file(self.processing)
        config["size"] = 256
        self.labels_fixture.write_json(self.processing, config)
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.run_images()
        self.assertEqual(
            old, _json_file(self.f.outputs / "audit/step_07_manifest.json")
        )
        _verify_run(self.f.outputs, 7)

    def test_source_change_during_processing_stops_publication(self):
        self.prepare()
        self.run_images()
        original = process_dataset
        old = _json_file(self.f.outputs / "audit/step_07_manifest.json")

        def changed(dataset, settings):
            result = original(dataset, settings)
            path = self.sources[100]
            path.write_bytes(path.read_bytes() + b"changed")
            return result

        with patch("src.imaging.prepare_images.process_dataset", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "changed during processing"):
                self.run_images()
        self.assertEqual(
            old, _json_file(self.f.outputs / "audit/step_07_manifest.json")
        )

    def test_readonly_input_validation(self):
        self.prepare()
        rows, index, context = load_inputs(self.root, self.configs)
        self.assertEqual(len(rows), len(index))
        self.assertEqual(
            context["source_label_run_id"],
            _json_file(self.labels_fixture.audit / "step_06_run_metadata.json")[
                "run_id"
            ],
        )

    def test_review_tail_extra_fields_and_changed_processing_hash(self):
        self.prepare()
        self.run_images()
        tail_path = self.f.outputs / "audit/step_07_csv_tail_checks.csv"
        with tail_path.open(encoding="utf-8", newline="") as stream:
            self.assertEqual(list(csv.DictReader(stream))[0]["status"], "no_csv_inputs")
        reviews = self.review_file({})
        complete = reviews.read_text(encoding="utf-8")
        lines = complete.splitlines()
        reviews.write_text(
            "\n".join(lines[:-1]) + "\ntruncated,record\n", encoding="utf-8"
        )
        rows = self.run_images(reviews)
        self.assertEqual(sum(r["review_complete"] for r in rows), 2)
        with tail_path.open(encoding="utf-8", newline="") as stream:
            tail = list(csv.DictReader(stream))[0]
        self.assertEqual(tail["status"], "incomplete")
        self.assertEqual(tail["skipped_rows"], "1")
        old = _file_hash(self.f.outputs / "audit/step_07_manifest.json")
        lines[1] += ",unexpected"
        reviews.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "extra fields"):
            self.run_images(reviews)
        reviews.write_text(complete, encoding="utf-8")
        config = _json_file(self.processing)
        config["preview_size"] = 512
        self.labels_fixture.write_json(self.processing, config)
        with self.assertRaisesRegex(ValueError, "Stale image review"):
            self.run_images(reviews)
        self.assertEqual(
            old, _file_hash(self.f.outputs / "audit/step_07_manifest.json")
        )
        _verify_run(self.f.outputs, 7)


if __name__ == "__main__":
    unittest.main()
