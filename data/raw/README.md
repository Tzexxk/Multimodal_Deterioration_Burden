# Raw Data

## Purpose

`data/raw/` documents the raw-data layer of this project. MIMIC data used by the project must not be stored in or distributed through the Git repository.

Raw data must remain in an authorized local or controlled storage location. Processing scripts locate the data through the machine-specific configuration file:

```text
configs/paths.local.json
```

This file contains real local paths and must not be committed to Git.

## Required data sources

The core bimodal stage requires:

```text
MIMIC-IV/
├── hosp/
│   ├── patients.csv
│   ├── admissions.csv
│   ├── labevents.csv
│   ├── d_labitems.csv
│   ├── transfers.csv
│   └── services.csv
└── icu/
    └── icustays.csv
```

```text
MIMIC-CXR/
├── cxr-record-list.csv
├── cxr-study-list.csv
└── dataset/
    └── ... DICOM files
```

The optional trimodal text-extension stage may also require:

```text
MIMIC-IV-NOTE/
└── note/
    └── radiology.csv
```

Actual filenames and directory layouts depend on the authorized dataset version available to each member.

## Local path configuration

First copy:

```text
configs/paths.example.json
```

and create the following local file:

```text
configs/paths.local.json
```

Example:

```json
{
  "mimic_iv": "SET_YOUR_MIMIC_IV_PATH",
  "mimic_cxr": "SET_YOUR_MIMIC_CXR_PATH",
  "mimic_note": "SET_YOUR_MIMIC_NOTE_PATH",
  "processed": "data/processed",
  "outputs": "outputs"
}
```

Replace the first three values with valid local paths.

Do not place personal drive letters, usernames, network-storage addresses, access tokens, or other credentials in shared configuration files, source code, or notebooks.

## Mandatory rules

1. Raw data must remain read-only and must not be modified, overwritten, or renamed.
2. Do not copy raw MIMIC files into the repository’s `data/raw/` directory.
3. Do not commit CSV files, DICOM files, patient-level indexes, report text, or other restricted data.
4. Do not expose patient-level data through Git, public cloud storage, public messages, screenshots, or uncleared notebook outputs.
5. Only members with the required data-access authorization may access or process the raw data.
6. All data use and sharing must comply with applicable data-access agreements, course requirements, and institutional policies.
7. Production code must locate raw data through `configs/paths.local.json`. Personal absolute paths must not be hard-coded.
8. Indexes, cohorts, features, and audit results derived from the raw data must be written to:

```text
data/processed/
outputs/
```

These derived-data directories are excluded from Git by default. Only aggregate results or version records that have been confirmed not to contain patient-level information and explicitly approved for sharing may be committed to their designated shared directories.

## Files permitted in this directory

This directory is used only for raw-data documentation and must not contain actual data:

```text
data/raw/
└── README.md
```

Do not create any of the following in this directory:

- copies of the raw data;
- shortcuts to the raw data;
- symbolic links;
- Windows directory junctions;
- file manifests containing patient identifiers.

Project code must always access raw data outside the repository through the local configuration file.

## Data validation

Before formal processing begins, run the project’s raw-data audit stage to verify that:

- all required files are present;
- all required columns can be read;
- local DICOM files exist and can be decoded;
- CXR index records correspond to the locally available image files;
- required time fields exist and use valid formats;
- raw-data directories remain read-only.

If validation fails, correct the local path, data download, or data-preparation problem. Do not modify the raw files to bypass an audit failure.
