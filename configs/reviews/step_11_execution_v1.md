# Step 11: portable training-package export

The approved export release is `burden_v2_h72_r01`. It binds the frozen
72-hour cohort, `burden_v2_h72_r01_split_v1`, `lab_dictionary_v1` and
`ehr_features_v1`. The common cohort contains 473 studies, 303 admissions and
222 patients. Study splits remain 337/69/67; patient splits remain 156/33/33.

## Run the export

From the repository root in the project's Python environment:

```powershell
python scripts/11_export_training_package.py
```

The user runs the full-data export. Codex runs synthetic tests and inspects
actual artifacts read-only. This command does not scan raw clinical CSVs,
decode DICOMs, transform pixels, resplit patients or fit model preprocessing.
It verifies frozen provenance, source versions, prepared NPY contents and
hashes before consuming those existing artifacts.

The independent `configs/training_export_v1.json` selects the identities.
Use `--export-config` for a deliberately approved new release configuration.
The existing path/config arguments remain available; keep input configurations
outside generated directories. No force overwrite or temporary rule overrides
are supported.

## Outputs

The formal directory is `data/processed/training/burden_v2_h72_r01/`:

- `samples.parquet`: only V1 required fields, exact dtypes, one row per study,
  sorted by sample_id. Labels and patient assignments remain unchanged.
- `ehr_features.parquet`: byte-for-byte copy of the frozen raw EHR wide table.
- `feature_schema.json`: byte-for-byte copy of the frozen whitelist/schema.
- `dataset_metadata.json`: interface and release versions, processing rules,
  source/configuration hashes, actual counts and package member checksums.
- `summary.csv`: long-form split/metric/value/unit/denominator rows recomputed
  from exported samples. Score bins 0-8 include zeros. Patient score bins use
  each patient's maximum; event summaries distinguish studies and patients.
- `README.md`: English loading example, generation time, limitations and model
  preprocessing responsibilities.
- `images/`: exactly the 473 included NPY files in their existing layout.

The additional archive is `data/processed/training/burden_v2_h72_r01.zip`.
It contains the same directory under one top-level release folder. Entry order,
timestamps, permissions and compression are fixed. ZIP entries are checked
against the exported member hashes, including the metadata file.

The local image manifest is
`data/processed/imaging/burden_v2/image_manifest.parquet`. It records source
DICOM path/hash, source NPY path/hash, image identity, t0, QC, processing hashes
and the joined split. Its frozen copy remains in the local audit release.
This manifest and all machine-specific source paths stay outside the package.

Audit records are under `outputs/audit/burden_v2/training/`:

- `training_package_freeze.json`: current immutable export record.
- `training_package_releases.json`: historical version/checksum registry.
- `step_11_run_metadata.json`: latest export or verification run.
- `releases/burden_v2_h72_r01/`: original freeze record and image manifest.

The shared transaction manifest and CSV-input status are
`outputs/audit/step_11_manifest.json` and `step_11_csv_tail_checks.csv`.
Step 11 reports `no_csv_inputs`; earlier source CSV checks remain bound through
upstream manifests. This does not claim that supplied source files are complete.

## Integrity and reruns

Every included image must be accepted by QC, match cohort identity, and be
float32, 224x224, finite and in [0,1]. Copying preserves NPY bytes; source and
destination hashes are checked before/after copying. Missing, damaged or
changing inputs block publication without dropping or repairing samples.

All three models share the same samples and split. EHR contains sample_id and
152 numeric plus five categorical features only. Numeric NaN and categorical
`__MISSING__` remain unchanged. Labels, split and observation fields do not
become EHR predictors.

The package uses relative `/` image paths. Changing DATA_ROOT after moving or
extracting the package is sufficient for loading. There are no DICOM paths or
personal drive paths in public metadata. Full local bindings stay in the audit
layer. Metadata does not hash itself or its containing ZIP; the external freeze
record binds both, avoiding checksum cycles.

Identical reruns verify and reuse the original directory, ZIP and generation
time. Changed bound sources, code or configuration cannot overwrite that release.
A new approved release preserves previous versions. Failed/interrupted
publication restores the prior files; do not delete recovery journals or
backups. Empty staging directories from an unsuccessful first export do not
prevent a retry.

Successful export sets `training_package_exported=true` and leaves
`independent_validation_completed=false`. Step 12 independently validates the
actual package and relocation before final acceptance. Do not treat Step 11
export checks as independent Step 12 completion.

## Synthetic checks

```powershell
python -m unittest discover -s tests/data
python -m unittest discover -s tests/ehr
python -m ruff check .
python scripts/11_export_training_package.py --help
git diff --check
```

Synthetic cases cover interface types, labels, patient isolation, whitelist,
image identity/content, file changes, ZIP inventory/checksums, relocation,
immutable versions, failed publication and actual process-death recovery.
No new dependencies, commits, pushes, uploads or external sharing are performed.
