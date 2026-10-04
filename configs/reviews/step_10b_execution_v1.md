# Step 10B: approved features and execution

The approved laboratory dictionary is `lab_dictionary_v1`. Its 25 retained
items produce six aggregates each, plus two numeric static features and five
categorical static features. The resulting table has 152 numeric features,
five categorical features and `sample_id`. All 473 frozen cohort keys remain.
The 72-hour outcome horizon and patient assignments are unchanged.

## Run the audit first

From the repository root, using the project's Python environment:

```powershell
python scripts/10b_build_ehr_features.py --mode audit `
  --dictionary-version lab_dictionary_v1
```

The user runs the full-data pipeline. Codex runs synthetic tests and verifies
actual artifacts read-only. Audit does not publish the final EHR wide table.

Audit products are under `outputs/audit/burden_v2/ehr/ehr_audit_v1/`:

- `ehr_audit_manifest.json`: source, code, dictionary, cohort, split and
  configuration bindings; checksums; blocking conflict count; readiness.
- `ehr_audit_report.md`: limitations and completion state.
- `static_conflicts.csv` and `blocking_conflicts.csv`: explicit unresolved
  identity, value, time or laboratory-rule conflicts. Empty reports retain headers.
- `lab_missingness.csv`: pre-cleaning and post-cleaning study coverage, patient
  coverage and denominators for all/train/validation/test. Coverage before
  cleaning counts eligible finite valuenum records, matching candidate selection.
- `ehr_feature_summary.csv`: column missingness, ranges, distinct values,
  all-missing and constant flags. Validation/test summaries are descriptive only.
- `lab_cleaning_summary.csv`: revision and numeric rejection dispositions with
  source-record and study-association counts kept separate.
- `lab_window_rejections.csv`: availability and admission-ownership failures.
- `feature_schema_candidate.json`: exact candidate whitelist and feature order.

Bound caches are under `data/cache/burden_v2/ehr/ehr_audit_v1/`. They include the
candidate table, the complete retained-item cleaning trace, and filtered static
source subsets. The trace contains sample/source keys, timestamps, raw units,
specimens, numeric values, assignment provenance and revision dispositions.
Invalid/missing charttimes remain documented in the frozen Step 10A audit.

All approved items remain even when cleaned coverage falls across a previous
30%, 50% or 80% missingness band. Inspect train all-missing, constant and changed
coverage flags before building. No automated feature selection uses any split.

Ordinary static missingness is retained as NaN or `__MISSING__`. Conflicting
identities, invalid nonempty values, differing latest service values and
inconsistent admission times block build. Resolve these through review and a
new approved audit configuration/version; do not edit frozen dictionaries or
cached files. A successful command can publish a blocked audit: inspect
`audit_checks_passed` and `blocking_conflicts`, not only the process exit code.

## Build after read-only audit verification

Only after the audit has passed and its actual outputs have been checked:

```powershell
python scripts/10b_build_ehr_features.py --mode build `
  --dictionary-version lab_dictionary_v1 `
  --feature-set-version ehr_features_v1
```

The build consumes the bound audit cache and checks sources again. It publishes:

- `data/processed/ehr/burden_v2/ehr_features_raw.parquet`
- `data/processed/ehr/burden_v2/feature_schema.json`
- immutable copies and `ehr_features_freeze.json` under
  `data/processed/ehr/burden_v2/releases/ehr_features_v1/`
- the current freeze record in the audit directory and the historical registry
  `outputs/audit/burden_v2/ehr/ehr_features_releases.json`.

Feature-set versions cannot be overwritten with changed bindings. Identical
reruns verify and reuse the frozen snapshot. Audit versions also bind exact
inputs and rules: changed sources, code or configuration require a new
`audit_version` in an independent feature configuration. Do not delete recovery
journals or backups to force reruns.

The shared publication helper uses independent namespaces:
`outputs/step_10b/audit/audit/step_10_manifest.json` and
`outputs/step_10b/build/audit/step_10_manifest.json`. Each also records CSV-tail
checks. The numeric helper step identifier is 10, but these directories are
independent of the original Step 10A manifest and recovery journal.

## Definitions and boundaries

Each lab yields `last_24h`, `min_24h`, `max_24h`, `mean_24h`, `count_24h` and
`hours_since_last`, prefixed by its approved feature name. Sampling must fall
within inclusive `[t0-24h,t0]`, with `charttime <= storetime <= t0`. Admission
ownership and missing-admission inference reuse the frozen audit rules.

Revision groups use patient, admission, item, specimen and charttime. Choose
the latest available storetime, then the greatest numeric labevent ID and
stable source key. Select revisions before numeric filtering; an invalid latest
revision does not revive a superseded value. Distinct specimens remain distinct.
Results may legitimately contribute to multiple study windows.

Count is the number of cleaned results. With no valid measurement, count is
zero and other lab aggregates are NaN. Recent value ties use charttime,
storetime, numeric record ID and source key. Hours since last uses charttime.
Finite extremes are preserved; reference ranges do not define validity bounds.
The approved MCHC branches and INR missing-unit handling are applied exactly.
Double-precision aggregation precedes float32 storage; infinities and float32
overflow stop publication.

Age and careunit reuse confirmed frozen cohort evidence. Sex comes from
patients.gender; admission type/location come from the same admission.
Current service uses the latest same-patient/admission `transfertime <= t0`.
Missing latest service values stay missing; no fallback to an earlier service.
Hours from admission uses the confirmed admittime. No future treatment,
discharge, outcome, identifier or split column enters the feature whitelist.

Service availability uses transfertime because services supplies no storetime.
Deidentified anchor ages cannot recover actual ages above 89. These limitations
are recorded in the schema and report. Whitespace is trimmed from static
categories; categories are otherwise preserved without grouping or vocabularies.

Numeric missing values are float32 NaN; categorical missing values are
`__MISSING__`. No filling, scaling, clipping, missing-indicator expansion or
model fitting occurs here. Those operations remain train-only at the model
stage. `ehr_features_ready=true` means local feature publication passed;
portable export and independent validation remain Steps 11-12.

## Synthetic verification

```powershell
python -m unittest discover -s tests/ehr
python -m unittest discover -s tests/cohort
python -m ruff check .
python scripts/10b_build_ehr_features.py --help
git diff --check
```

No commits, pushes, new dependencies, production reruns by Codex or edits to
Step 10A frozen code/configuration are part of this stage.
