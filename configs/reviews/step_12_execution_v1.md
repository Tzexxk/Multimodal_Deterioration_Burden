# Step 12: independent validation and acceptance

Run from the repository root in the project's Python environment:

```powershell
python scripts/12_validate_training_package.py
if ($LASTEXITCODE -ne 0) { throw "Step 12 did not accept the training package. Inspect the reports." }
```

The user runs this complete validation. Codex runs synthetic tests and checks
actual reports read-only afterward. This command does not rescan complete raw
clinical CSVs, alter clinical rules, rebuild features, transform images or fit
a model. It reads the bound audit subsets, frozen tables and source images.

The independent `configs/training_validation_v1.json` selects the existing
release, cohort, split, laboratory dictionary and EHR feature identities.
The frozen 72-hour horizon, 473 studies, 303 admissions, 222 patients and
337/69/67 study splits remain unchanged. Patient splits remain 156/33/33.
Existing path/config arguments are available; use `--validation-config` only
for a separately approved validation configuration. There is no skip, force
overwrite or repair option.

## Independent checks

- Verify completed Steps 01-11 and Step 10B audit/build manifests, immutable
  snapshots, registries, configuration/code hashes and source versions.
- Check exact V1 sample types, unique study/image keys, patient assignments,
  cohort values, the domain-capped burden formula and observation/event times.
- Derive the EHR whitelist from the frozen reviewed dictionary. Check feature
  order, float32 numeric values with NaN, string categories and `__MISSING__`.
- Stream the bound uncleaned laboratory subset in batches. Independently
  establish the inclusive 24-hour charttime window, storetime availability,
  patient/admission ownership and unique inference using all patient admissions.
  Select revisions before numeric rejection; do not backfill an older numeric
  result. Recompute all six aggregates using the approved rules. The cleaning
  trace is not used as the source of truth. References do not become bounds.
- Reconstruct static features from bound patient/admission/service subsets
  and frozen known-at-t0 cohort evidence. Services use transfertime as the
  availability proxy, including conflict and missing-latest handling.
- Check all 473 image keys, QC, source DICOM/NPY hashes and portable NPY
  arrays: float32, 224 by 224, finite and within [0, 1].
- Recompute summaries and verify metadata/settings and member checksums.
- Inspect ZIP inventory, paths, entry types, encryption and expected
  decompressed sizes before extraction. Extract verified members into an owned
  temporary directory, align all EHR rows and targets, load all images using
  only the relocated DATA_ROOT, then remove the temporary directory on success
  or failure. No clinical temporary directory is retained.
- Recheck evidence hashes, source versions and source image hashes after
  validation. Model preprocessing and validation-threshold checks remain
  `not_applicable` until models are developed.

Business checks do not call Step 10B aggregation/static validators or Step 11
sample/summary/archive validators. Shared helpers are limited to provenance
hashing, safe paths and recoverable publication. The validator supports the
approved V1 selectors; a changed selector blocks acceptance pending a deliberate
validator update. Numeric aggregates compare after float32 conversion; the
mean permits one float32 ULP for independent summation order.

## Reports and acceptance

Current reports are outside the immutable training package:

- `outputs/audit/burden_v2/leakage_checks.json`
- `outputs/audit/burden_v2/cross_file_checks.csv`
- `outputs/audit/burden_v2/training_package_acceptance.json`
- `outputs/audit/burden_v2/step_12_run_metadata.json`
- `outputs/audit/step_12_manifest.json`

Each run also preserves JSON/Markdown reports, the acceptance record,
cross-file checks and bounded `failure_details.csv` under:

```text
outputs/audit/burden_v2/validation/training_validation_v1/burden_v2_h72_r01/<run_id>/
```

Statuses are `pass`, `fail`, `blocked` or `not_applicable`. Every mandatory
check must pass. Missing evidence blocks validation; changed evidence or
invalid contents fail it. Each failing check stops at its first diagnostic.
Successful checks report zero affected rows; failed/blocked totals are unknown
and remain blank rather than being reported as zero. Failure details retain
headers when empty. The diagnostic limit bounds the number of recorded checks,
not an assertion that all affected records have been enumerated.

A successful current acceptance record has `independent_validation_completed`,
`validation_passed` and `training_package_accepted` set to true. Its binding
includes the exact package/ZIP, upstream evidence, validator code and validation
configuration hashes. Acceptance applies only while those bindings match.

A failed run publishes a rejected current record and exits nonzero, preserving
historical records. A completed Step 12 manifest means report publication
completed; it does not mean the package passed validation. Interrupted report
publication uses the existing recovery journal on the next run. Do not delete
journals or backups to bypass recovery.

The frozen Step 11 metadata and ZIP retain
`independent_validation_completed: false` as historical export facts. The
external Step 12 record is authoritative for final acceptance; do not rewrite
the immutable package to update that flag. No commits or pushes are performed.

After the user runs Step 12, Codex verifies the actual reports read-only.
Accepted data delivery completes Steps 10-12. Model development, train-fitted
preprocessing, validation thresholds and independent model evaluation are
subsequent work.
