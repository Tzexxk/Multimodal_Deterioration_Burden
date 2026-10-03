# ICU Risk

## Configuration for the raw data audit

Configuration files use standard JSON. Do not add `//` or `/* ... */` comments to
them: Python's `json` parser does not accept comments. This section explains which
settings depend on the computer or local copy of the data.

Copy `configs/paths.example.json` to `configs/paths.local.json`, then set the raw
data roots for your computer. `paths.local.json` is excluded from Git. Raw MIMIC
data must remain read-only.

| File and setting | When to adjust it |
| --- | --- |
| `paths.local.json`: `mimic_iv`, `mimic_cxr` | Required absolute paths to your local MIMIC-IV and MIMIC-CXR roots. |
| `paths.local.json`: `mimic_note` | Set an absolute path if MIMIC-IV-NOTE is available. The Step 01 audit treats the radiology file as optional. |
| `paths.local.json`: `outputs` | Usually leave as `outputs`. A relative path is resolved from the repository root; keep it outside every raw data root and in controlled storage. |
| `paths.local.json`: `processed` | Usually leave as `data/processed`. Step 01 does not use this path. |
| `imaging.json`: `raw_audit.dicom_directory` | Change only if your DICOM directory has a different name or location under `mimic_cxr`. The default is `dataset`. |
| `imaging.json`: `raw_audit.dicom_path_layout` | Leave as `auto` for either supported layout. Use `standard` for `dataset/files/pXX/p<subject_id>/s<study_id>/...`, or `flattened` for `dataset/p<subject_id>/s<study_id>/...`. An explicit layout can avoid fallback path checks. |
| `imaging.json`: `raw_audit.dicom_workers`, `raw_audit.dicom_batch_size` | Tune for local CPU, memory, and storage. Defaults are 8 workers and 128 index rows per batch. The batch size must be at least the worker count. Compare the `tqdm` image rate between runs. |
| `imaging.json`: `raw_audit.record_list`, `raw_audit.study_list`, `raw_audit.dicom_suffixes` | Change only if the local data release uses different file names or DICOM suffixes; record the input version before comparing audit results. |

The remaining audit settings express project rules rather than machine preferences.
Keep pixel decoding enabled for a complete readability audit. Changes to frontal
views, acquisition-time priority, required EHR columns, time fields, or EHR sample
size should be agreed upon and versioned by the team.

Run Step 01 from the repository root with the project Python environment:

```powershell
python scripts\01_audit_data.py
```

The script writes six audit files under `outputs/audit/` and a timestamped run log
under `outputs/logs/`. The audit scans the local DICOM directory on each run and
does not depend on files found on another computer. Empty indices and zero local
image coverage return exit code `2`; partial downloads can pass when indexed
local images are available and the other audit checks complete.

## Running Steps 01–04

Run the commands in order from the repository root, using the project environment:

```powershell
python scripts/01_audit_data.py
python scripts/02_build_cxr_index.py
python scripts/03_select_frontal_image.py
python scripts/04_match_admissions.py
```

Every step stages its output files before publishing them. Its
`outputs/audit/step_XX_manifest.json` records the run status and SHA256 hashes of
the published files. Downstream steps check the relevant upstream manifest and
reject incomplete or modified outputs. Step 01 can publish a complete set of
diagnostic files with status `incomplete` and exit code `2`; inspect its report
and correct the inputs before continuing. Regenerate old outputs without
manifests, or outputs moved to another location, by rerunning the producing steps.

Wait for a step to finish before opening or using its outputs. Ordinary errors
and Ctrl+C during publication roll back the whole output set. If the process is
forcibly terminated during publication, `.step_XX_pending.json` and sibling backup
files preserve the recovery information, and downstream steps refuse to proceed.
After confirming that the previous process has stopped, rerun the interrupted
step: it restores the previous output set before starting the new run. If Windows
reports a file access error, close any viewer holding the output file and rerun.
Do not delete pending journals or backups manually. Run only one process per step.

All supported DICOM layouts validate the CSV path against its patient, study, and
image IDs. Duplicate `hadm_id` values anywhere in the admissions CSV stop Step 04,
including identical duplicate rows. Acquisition timestamps preserve embedded UTC
offsets, or use a valid DICOM `TimezoneOffsetFromUTC` when the timestamp has no
embedded offset. Missing offsets remain unspecified; malformed required offsets
make the image time unreliable. Incomparable image and admission times are not
forced into a unique match.

Run the synthetic regression checks without creating Python bytecode caches:

```powershell
python -B -m unittest discover -s tests/cohort -p test_pipeline_safety.py -v
```

These checks use temporary synthetic data and a simulated DICOM reader. They do
not validate the installed pixel decoder plugins or scan the real datasets.
