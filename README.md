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
under `outputs/logs/`. A rerun replaces the six audit files; wait for the run to
finish before using them. The audit scans the local DICOM directory on each run
and does not depend on files found on another computer.
