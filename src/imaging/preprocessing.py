"""A single deterministic DICOM-to-grayscale transformation for all image models."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import warnings

import numpy as np
from PIL import Image
from pydicom.dataset import Dataset
from pydicom.pixels import (
    apply_modality_lut,
    apply_presentation_lut,
    apply_voi,
    apply_windowing,
)
from pydicom.uid import (
    DigitalXRayImageStorageForPresentation,
    DigitalXRayImageStorageForProcessing,
)


class PixelIssue(ValueError):
    """A per-image failure, or metadata uncertainty that must be reviewed."""

    def __init__(self, code: str, *, pending: bool = False):
        super().__init__(code)
        self.pending = pending


def validate_settings(settings: dict) -> None:
    expected = {
        "size": 224,
        "percentiles": [1, 99],
        "percentile_method": "linear",
        "interpolation": "bilinear",
        "mask_interpolation": "nearest",
        "padding_value": 0,
        "rounding": "half_up",
    }
    if any(
        settings.get(k) != v or type(settings.get(k)) is not type(v)
        for k, v in expected.items()
    ):
        raise ValueError("Unsupported V2 image processing configuration")
    if (
        not isinstance(settings.get("processing_version"), str)
        or not settings["processing_version"].strip()
    ):
        raise ValueError("Missing image processing version")
    for key, low, high in (
        ("preview_size", 224, 2048),
        ("review_seed", 0, 2**32 - 1),
        ("review_per_group", 1, 100),
    ):
        if type(settings.get(key)) is not int or not low <= settings[key] <= high:
            raise ValueError(f"Invalid {key}")


def array_hash(pixels: np.ndarray) -> str:
    values = np.ascontiguousarray(
        pixels.astype(pixels.dtype.newbyteorder("<"), copy=False)
    )
    header = json.dumps(
        {"shape": list(values.shape), "dtype": values.dtype.str}, sort_keys=True
    ).encode()
    return hashlib.sha256(header + b"\0" + values.tobytes()).hexdigest()


def _lut_valid(
    item: Dataset, *, presentation: bool = False, dx_voi: bool = False
) -> bool:
    try:
        descriptor = list(item.LUTDescriptor)
        if len(descriptor) != 3 or any(type(v) is not int for v in descriptor):
            return False
        count, first, bits = descriptor
        count = count or 65536
        # PS3.3 C.8.11.3.1.5 specializes DX VOI entries to 10-16 bits.
        depths = range(8, 17) if presentation else range(10, 17) if dx_voi else (8, 16)
        if not 1 <= count <= 65536 or bits not in depths:
            return False
        if presentation and first != 0:
            return False
        data = item.LUTData
        if isinstance(data, bytes):
            lengths = (
                {count * 2} if bits > 8 else {count, count + (count % 2), count * 2}
            )
            return len(data) in lengths
        values = np.asarray(data)
        return (
            values.size == count
            and values.ndim == 1
            and bool(np.isfinite(values).all())
            and bool(((values >= 0) & (values < 2**bits)).all())
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _unit_preview(values: np.ndarray, valid: np.ndarray) -> np.ndarray | None:
    finite = valid & np.isfinite(values)
    if not finite.any():
        return None
    lo, hi = float(values[finite].min()), float(values[finite].max())
    if hi <= lo:
        return None
    output = np.zeros(values.shape, dtype=np.float32)
    output[finite] = np.clip((values[finite] - lo) / (hi - lo), 0, 1)
    return output


def _display_dataset(dataset: Dataset) -> Dataset:
    # Copy display metadata only, avoiding cached decoded arrays in Dataset.copy().
    result = Dataset()
    for name in (
        "PhotometricInterpretation",
        "BitsStored",
        "PixelRepresentation",
        "ModalityLUTSequence",
        "RescaleSlope",
        "RescaleIntercept",
        "VOILUTSequence",
        "WindowCenter",
        "WindowWidth",
        "VOILUTFunction",
        "PresentationLUTSequence",
        "PresentationLUTShape",
    ):
        if name in dataset:
            result.add(copy.deepcopy(dataset.data_element(name)))
    if hasattr(dataset, "file_meta"):
        result.file_meta = copy.deepcopy(dataset.file_meta)
    result.set_original_encoding(*dataset.original_encoding)
    return result


def _display(
    values: np.ndarray, valid: np.ndarray, ds: Dataset, meta: dict
) -> np.ndarray:
    working = _display_dataset(ds)
    details = {
        name: str(ds.get(name, ""))
        for name in (
            "RescaleSlope",
            "RescaleIntercept",
            "WindowCenter",
            "WindowWidth",
            "VOILUTFunction",
            "PresentationLUTShape",
            "PixelPaddingValue",
            "PixelPaddingRangeLimit",
        )
    }
    meta["display_metadata"] = details
    issues = meta["issues"]
    modality = working.get("ModalityLUTSequence", [])
    selected = next((i for i, item in enumerate(modality) if _lut_valid(item)), None)
    if selected is not None:
        meta["modality_path"] = "modality_lut"
        details["modality_lut_index"] = selected
        details["modality_lut_descriptor"] = list(modality[selected].LUTDescriptor)
        if selected or len(modality) != 1:
            issues.append("nonstandard_modality_lut_sequence")
        working.ModalityLUTSequence = [modality[selected]]
        # Windowing must not incorporate a rescale transform superseded by the LUT.
        for name in ("RescaleSlope", "RescaleIntercept"):
            if name in working:
                del working[name]
    else:
        if "ModalityLUTSequence" in working:
            issues.append("invalid_modality_lut_fallback")
            del working.ModalityLUTSequence
        if "RescaleSlope" in working or "RescaleIntercept" in working:
            try:
                slope = float(working.get("RescaleSlope", 1))
                intercept = float(working.get("RescaleIntercept", 0))
                if not np.isfinite([slope, intercept]).all():
                    raise ValueError()
            except (TypeError, ValueError):
                raise PixelIssue("invalid_rescale", pending=True) from None
            working.RescaleSlope, working.RescaleIntercept = slope, intercept
            details.update(
                applied_rescale_slope=slope, applied_rescale_intercept=intercept
            )
            meta["modality_path"] = "rescale"
        else:
            meta["modality_path"] = "identity"
    try:
        values = apply_modality_lut(values, working)
    except Exception as error:
        raise PixelIssue(
            f"modality_lut_error:{type(error).__name__}", pending=True
        ) from error
    if not np.isfinite(values[valid]).all():
        raise PixelIssue("nonfinite_modality_values")
    voi = working.get("VOILUTSequence", [])
    dx_voi = ds.get("SOPClassUID") in {
        DigitalXRayImageStorageForPresentation,
        DigitalXRayImageStorageForProcessing,
    }
    selected = next(
        (i for i, item in enumerate(voi) if _lut_valid(item, dx_voi=dx_voi)), None
    )
    if selected is not None:
        details["voi_lut_index"] = selected
        details["voi_lut_descriptor"] = list(voi[selected].LUTDescriptor)
        if selected:
            issues.append("invalid_voi_lut_skipped")
        meta["voi_path"] = "voi_lut"
        # pydicom warns for fractional LUT indices; retain the result for review.
        if (
            np.issubdtype(values.dtype, np.floating)
            and np.equal(values[valid], np.floor(values[valid])).all()
        ):
            values = values.astype(np.int64)
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            try:
                values = apply_voi(values, working, index=selected)
            except Exception as error:
                raise PixelIssue(
                    f"voi_lut_error:{type(error).__name__}", pending=True
                ) from error
        if observed:
            issues.append("voi_lut_numeric_warning")
    else:
        if "VOILUTSequence" in working:
            issues.append("invalid_voi_lut_fallback")
        center, width = working.get("WindowCenter"), working.get("WindowWidth")
        if center is not None and width is not None:
            centers = list(center) if hasattr(center, "__iter__") else [center]
            widths = list(width) if hasattr(width, "__iter__") else [width]
            function = str(working.get("VOILUTFunction", "LINEAR")).strip().upper()
            selected = None
            for index, (center, width) in enumerate(zip(centers, widths)):
                try:
                    center, width = float(center), float(width)
                    if (
                        function not in {"LINEAR", "LINEAR_EXACT", "SIGMOID"}
                        or not np.isfinite([center, width]).all()
                        or width < (1 if function == "LINEAR" else np.nextafter(0, 1))
                    ):
                        continue
                    working.WindowCenter, working.WindowWidth = center, width
                    windowed = apply_windowing(values, working, index=0)
                except Exception:
                    continue
                selected = index
                values = windowed
                details.update(
                    window_index=index,
                    applied_window_center=center,
                    applied_window_width=width,
                    applied_window_function=function,
                )
                meta["voi_path"] = "window"
                break
            if selected is None:
                issues.append("invalid_window_no_voi")
                meta["voi_path"] = "none"
            elif selected or len(centers) != len(widths):
                issues.append("invalid_window_group_skipped")
        else:
            if center is not None or width is not None:
                issues.append("incomplete_window_no_voi")
            meta["voi_path"] = "none"
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values[valid]).all():
        raise PixelIssue("nonfinite_voi_values")
    if "PresentationLUTSequence" in working:
        seq = working.PresentationLUTSequence
        if (
            len(seq) != 1
            or not _lut_valid(seq[0], presentation=True)
            or "PresentationLUTShape" in working
        ):
            raise PixelIssue("invalid_presentation_lut", pending=True)
        if values[valid].max() <= values[valid].min():
            raise PixelIssue("constant_display_values")
        # Padding does not define the presentation LUT's quantization range.
        values[~valid] = values[valid].min()
        try:
            values = apply_presentation_lut(values, working).astype(np.float64)
        except Exception as error:
            raise PixelIssue(
                f"presentation_lut_error:{type(error).__name__}", pending=True
            ) from error
        meta["presentation_path"] = "presentation_lut"
        details["presentation_lut_descriptor"] = list(seq[0].LUTDescriptor)
    else:
        shape = str(working.get("PresentationLUTShape", "")).strip().upper()
        expected = (
            "INVERSE"
            if meta["photometric_interpretation"] == "MONOCHROME1"
            else "IDENTITY"
        )
        if "PresentationLUTShape" in working and shape != expected:
            raise PixelIssue("presentation_polarity_conflict", pending=True)
        meta["presentation_path"] = "shape_" + shape.lower() if shape else "photometric"
        meta["inversion_applied"] = expected == "INVERSE"
        if meta["inversion_applied"]:
            values[valid] = values[valid].max() + values[valid].min() - values[valid]
    return values


def process_dataset(dataset: Dataset, settings: dict) -> dict:
    """Return pixels/previews and audit metadata; never replace failed pixels with zeros."""
    meta = {
        "processing_status": "fail",
        "error_code": "",
        "issues": [],
        "photometric_interpretation": str(dataset.get("PhotometricInterpretation", ""))
        .strip()
        .upper(),
        "modality_path": "not_applied",
        "voi_path": "not_applied",
        "presentation_path": "not_applied",
        "inversion_applied": False,
        "before_method": "unavailable",
        "display_metadata": {},
        "pixels": None,
        "before": None,
    }
    try:
        try:
            values = dataset.pixel_array
        except Exception as error:
            raise PixelIssue(f"pixel_decode_error:{type(error).__name__}") from error
        meta["decoded_pixel_sha256"] = array_hash(values)
        if (
            int(dataset.get("NumberOfFrames", 1)) != 1
            or int(dataset.get("SamplesPerPixel", 1)) != 1
            or values.ndim != 2
        ):
            raise PixelIssue("unsupported_multiframe_or_color")
        if (
            values.shape != (int(dataset.Rows), int(dataset.Columns))
            or not np.issubdtype(values.dtype, np.number)
            or np.iscomplexobj(values)
        ):
            raise PixelIssue("unsupported_pixel_structure")
        meta.update(
            original_height=int(values.shape[0]), original_width=int(values.shape[1])
        )
        if meta["photometric_interpretation"] not in {"MONOCHROME1", "MONOCHROME2"}:
            raise PixelIssue("unsupported_photometric", pending=True)
        valid = np.ones(values.shape, dtype=bool)
        if "PixelPaddingValue" in dataset:
            try:
                low = float(dataset.PixelPaddingValue)
                high = float(dataset.get("PixelPaddingRangeLimit", low))
                if not np.isfinite([low, high]).all():
                    raise ValueError()
            except (TypeError, ValueError):
                raise PixelIssue("invalid_padding_metadata", pending=True) from None
            valid &= ~((values >= min(low, high)) & (values <= max(low, high)))
        elif "PixelPaddingRangeLimit" in dataset:
            raise PixelIssue("padding_range_without_value", pending=True)
        meta["valid_pixel_fraction"] = float(valid.mean())
        if not valid.any():
            raise PixelIssue("no_valid_pixels")
        if not np.isfinite(values[valid]).all():
            raise PixelIssue("nonfinite_pixels")
        meta["before"] = _unit_preview(values.astype(np.float64), valid)
        meta["before_method"] = (
            "raw storage values, min/max display; not a diagnostic rendering"
        )
        values = _display(values, valid, dataset, meta)
        if not np.isfinite(values[valid]).all():
            raise PixelIssue("nonfinite_display_values")
        meta["before"] = _unit_preview(values, valid)
        meta["before_method"] = (
            "DICOM display transforms, valid-pixel min/max; before percentile clipping and resize"
        )
        p1, p99 = np.percentile(
            values[valid], settings["percentiles"], method=settings["percentile_method"]
        )
        meta.update(p1=float(p1), p99=float(p99))
        if not np.isfinite([p1, p99]).all() or p99 <= p1:
            raise PixelIssue("constant_or_invalid_percentiles")
        meta["clipped_low_fraction"] = float((values[valid] < p1).mean())
        meta["clipped_high_fraction"] = float((values[valid] > p99).mean())
        normalized = np.zeros(values.shape, dtype=np.float32)
        normalized[valid] = np.clip((values[valid] - p1) / (p99 - p1), 0, 1)
        height, width = values.shape
        size = settings["size"]
        factor = min(size / height, size / width)
        rh, rw = (
            min(size, max(1, math.floor(v * factor + 0.5))) for v in (height, width)
        )
        resized = np.asarray(
            Image.fromarray(normalized).resize((rw, rh), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ).copy()
        mask = (
            np.asarray(
                Image.fromarray(valid.astype(np.uint8) * 255).resize(
                    (rw, rh), Image.Resampling.NEAREST
                )
            )
            > 0
        )
        resized[~mask] = 0
        if not mask.any() or np.ptp(resized[mask]) <= 0:
            raise PixelIssue("no_remaining_image_contrast")
        top, left = (size - rh) // 2, (size - rw) // 2
        output = np.zeros((size, size), dtype=np.float32)
        output[top : top + rh, left : left + rw] = resized
        meta.update(
            resized_height=rh,
            resized_width=rw,
            pad_top=top,
            pad_bottom=size - rh - top,
            pad_left=left,
            pad_right=size - rw - left,
        )
        validate_pixels(output, size)
        meta["pixels"] = output
        meta["processing_status"] = "complete"
    except PixelIssue as error:
        meta["processing_status"] = "pending_review" if error.pending else "fail"
        meta["error_code"] = str(error)
    except Exception as error:
        meta["processing_status"] = "pending_review"
        meta["error_code"] = f"unsupported_metadata:{type(error).__name__}"
    return meta


def validate_pixels(pixels: np.ndarray, size: int = 224) -> None:
    if (
        pixels.dtype != np.float32
        or pixels.shape != (size, size)
        or not np.isfinite(pixels).all()
        or pixels.min() < 0
        or pixels.max() > 1
    ):
        raise ValueError(
            "Invalid prepared image: expected finite float32 [224,224] in [0,1]"
        )
