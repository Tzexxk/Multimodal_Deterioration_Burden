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

The script writes audit files under `outputs/audit/` and a timestamped run log
under `outputs/logs/`. The audit scans the local DICOM directory on each run and
does not depend on files found on another computer. Empty indices and zero local
image coverage return exit code `2`; partial downloads can pass when indexed
local images are available and the other audit checks complete.

## Running Steps 01-05

Run the commands in order from the repository root, using the project environment:

```powershell
conda activate ICU_Risk
python scripts/01_audit_data.py
python scripts/02_build_cxr_index.py
python scripts/03_select_frontal_image.py
python scripts/04_match_admissions.py
python scripts/05_audit_burden_labels.py
```

Run each command only after the preceding step succeeds. You do not need to
delete existing outputs before rerunning. A successful run replaces that step's
published artifacts; a failed run preserves or restores the previous output set.
Timestamped run logs remain available. Do not delete recovery journals or backups.

## CSV final-record checks

Pipeline scripts use shared CSV checks implemented in
`src/cohort/common.py`. Each CSV input that a step reads or screens is checked
for a parseable final logical record with the same field count as its header.
Quoted fields, embedded newlines, UTF-8 BOM, LF/CRLF, and valid records without a
trailing newline are supported. A missing trailing newline alone is not an error.

The check first uses bounded input bytes. If the logical record boundary cannot
be established quickly, it reports `unverified`; the actual streaming reader
confirms the result when it reaches EOF. Sample-only readers can leave a large
input unverified. This avoids an extra full scan of large event or note tables.

Each step prints its findings and publishes
`outputs/audit/step_XX_csv_tail_checks.csv`. Its main audit report includes the
detail report name and counts of incomplete inputs, unverified checks and skipped
records. The detail report distinguishes `complete`, `incomplete`, `unverified`,
`invalid`, `unreadable` and `missing`. Steps 03, 06 and 08 parse no CSV records and
record `no_csv_inputs`. Step 07 also records `no_csv_inputs` unless `--reviews`
supplies an image-review CSV; that file uses the same final-record checks.
Step 06 hashes review inputs without parsing them and
references the Step 05 source checks. `skipped_rows` counts the distinct final record skipped by an
application reader; a sample-only screen can detect an incomplete tail without
encountering or skipping it.

For manual review, each detail row also includes `header_text` (the source header
with its CSV quoting preserved) and `last_line_text` (the last nonempty physical
line without its line ending). These fields remain available for `unverified`
inputs. A physical line can be only part of a quoted multiline record; compare
it with the header as a review aid, not proof of a complete logical record.
If that physical line exceeds the bounded preview, only its final 64 KiB are
shown and `last_line_truncated` is `True`. UTF-8 fragments split at the preview
boundary use replacement characters. CSV quoting preserves commas, quotes and
embedded header newlines when these previews are written to the report.

An incomplete final record produces a warning and is skipped; the program can
continue using the available valid records. Raw CSVs are never modified, and
missing fields are never padded. Only a confirmed incomplete record at EOF may
be skipped. Intermediate CSV errors, extra final fields, missing required schema
columns and other validation failures retain their existing error handling.
DuckDB rejects are checked against the original EOF bytes before publication.

If a report shows an incomplete CSV, manually review the file's course or other
source. A `complete` result establishes only the final record's structure; it
does not prove that all expected patients or event records were supplied. Audits
use the course data actually provided. In Step 05, a tail warning does not itself
make clinical labels unresolved or block window review. The existing dictionary,
clinical ambiguity and evidence rules still apply; an operational zero means no
qualifying event was found in the provided records, not proof that no event ever
occurred.

## Output publication and validation

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

## Step 05: V2 deterioration burden audit

Step 05 implements Sections 6-8 of
`docs/Data_Processing_and_Cleaning_Extension_V2.md`. It reads the validated
Step 03/04 tables and audits both 48-hour and 72-hour windows. Run it after
Steps 01-04 have finished for the current image subset:

```powershell
conda activate ICU_Risk
python scripts/05_audit_burden_labels.py
```

The optional `--paths`, `--cohort`, `--ehr` and `--imaging` arguments select
alternative JSON configuration files. Raw data stays read-only. Step 05 does
not decode images, choose H, freeze a cohort, split patients or train models.
If the image subset changes, rebuild Steps 01-04 first. Changed admissions or
selection inputs require a new Step 04 run.

Incomplete final CSV records are recorded in `step_05_csv_tail_checks.csv` and
skipped under the shared EOF policy. Review the file source if a warning appears.
Other CSV or schema errors preserve previous outputs. If only Step 05 event files
are replaced, rerun Step 05. Replacing admissions requires rerunning Step 04 first.

One sample represents one selected study, with its own acquisition time `t0`.
The audit keeps adults with estimated age
`anchor_age + year(t0) - anchor_year >= 18` and a unique admission match.
Current ICU/CCU status excludes a study; prior ICU does not. Unknown age or
care location and conflicting clinical times require review. Clinical timestamps
stay on the timezone-naive MIMIC timeline; offsets are never silently removed.

Each event window is `(t0, min(t0 + H, dischtime)]`. Events in the first three
hours count. Live discharge truncates follow-up, without erasing earlier events.
Six binary components form the audit target:

```text
T = max(new_icu_ccu_transfer, new_stepdown_transfer)
R = max(new_invasive_ventilation, new_severe_hypoxemia)
C = new_iv_vaso_inotrope_infusion
D = in_hospital_death
burden_score = T + 2*R + 2*C + 3*D
phys_death = int(R or C or D)
high_burden = int(burden_score >= 3)
```

The score ranges from 0 to 8. Pending component decisions stay null, and a
study receives a score only when eligibility and all six decisions are resolved.
An operational zero means no qualifying record under these rules, including
no ICU SpO2 record. It does not prove that a clinical event never occurred.

### Review the local dictionaries

`configs/cohort.json` contains `burden_v2` rules. All initial dictionary entries
are unconfirmed. The first run produces an inventory and pending cases for
review, so it can finish successfully without producing any assessable scores.
Never interpret a successful exit as clinical approval.

Start with `outputs/audit/burden_v2/dictionary_inventory.csv` and the component
frequency files. For each observed value, add or review its exact mapping in
the following configuration sections:

| Section | Mapping field and allowed values |
| --- | --- |
| `careunits` | `care_level`: `ward`, `stepdown_intermediate`, `icu_ccu`, `other`, `unknown`. Raw names are exact keys. ED, PACU and unmapped units are not assumed to be wards. |
| `transfer_event_actions` | `action`: `location` or `exclude`. Confirm local location records and exclude administrative discharge records. |
| `medications` | `canonical_drug`: one of the six configured drugs, or JSON `null` for a reviewed non-target drug. Keys are stripped, casefolded complete names. List aliases explicitly. |
| `administration_types` | `action`: `infusion`, `exclude` or `review`. A confirmed `review` policy keeps the administration method unresolved; it does not establish an infusion or a negative label. Blank detail fields are ignored when other details identify the event's method. IV bolus is excluded. |
| `medication_event_actions` | `action`: `start`, `restart`, `continue`, `stop`, `given`, `exclude` or `review`. A confirmed `review` policy leaves execution or onset unresolved and does not change established infusion state. Assign an onset action only when the source value documents it. |
| `procedure_status_actions` | `action`: `include` or `exclude`, based on actual recorded procedure status. |
| `iv_routes` | `action`: `iv`, `non_iv` or `missing` (blank route only). Route is a consistency check, not a substitute for administration type. Blank route fields stay missing and alone do not add review; explicit non-IV routes conflicting with target infusion evidence require review. |
| `spo2_units` | Confirm the local percent-unit values. |
| `items` | Confirm `225792`, `225794` and `220277`. Their labels and source tables are checked against local `d_items.csv` on every run. |

Every confirmed entry needs `confirmed: true`, a nonempty `reviewer` and
`mapping_reason`. For example, after reviewing a local unit:

```json
{
  "care_level": "ward",
  "confirmed": true,
  "reviewer": "reviewer_id",
  "mapping_reason": "Reviewed against the local care-unit definition"
}
```

Increment `dictionary_version` when mappings change, then rerun Step 05.
Unknown medication names are not assumed to be unrelated drugs. The inventory
also includes non-target source values, so its dictionary-completion gate is
conservative. Confirm the applicable values rather than copying guessed rules.

Care-unit intervals with an explicit matching admission ID can start before
admission or end after discharge. Their original times remain intact; location
checks use half-open intervals and event checks still use
`(t0, min(t0 + H, dischtime)]`. Crossing an admission boundary alone does not
require review. Confirmed administrative discharge records are excluded from
location checks and are not assigned an inferred outtime. Missing transfer
outtimes use the next confirmed location interval or discharge, with the basis
recorded. Missing admission IDs still require unique time-based ownership.
Invalid or uncertain intervals that may cover `t0` require candidate review;
future interval conflicts are reviewed in the affected event components.
The reviewer classified `Discharge Lounge`, `Emergency Department`,
`Emergency Department Observation` and `PACU` as `other` and excluded studies
located there at `t0`. Their confirmed mappings set `exclude_at_t0: true`;
the candidate reason is `excluded_careunit_at_t0`. This policy applies to the
study's current location, not to every study in the admission. Other `other`
care units are not excluded automatically. Conflicting simultaneous locations
remain subject to review. `careunit_dictionary.csv` reports the exclusion flag.
These exact-name policies were reviewed separately. The confirmed
`stepdown_intermediate` classification of `Observation` remains in effect.

Ventilation starts use direct `procedureevents` intervals. Overlapping and
touching intervals merge; continued support is not a new start. Airway or
ventilation-related `chartevents` are discovered for cross-checking only. Without
a corresponding direct episode they require review, including records whose
clinical meaning cannot be inferred from the item name.

SpO2 uses ICU item `220277` and the fixed threshold of 85%. Zero values, warnings,
unit uncertainty and simultaneous normal/low conflicts require review. Invalid
nonfinite or out-of-range values are excluded. Old measurements from an ended
ICU stay do not establish the current baseline.

eMAR details aggregate to one administration per `emar_id`. An explicit start or
restart can qualify; continued bags, rate changes, planned/cancelled doses and
boluses do not create a start. A first `given` record alone cannot establish onset.
The rules use neither an infusion-rate threshold to define treatment nor an
arbitrary time gap to infer that support stopped.
Confirmed excluded medications or administrations remain non-qualifying when
their timestamps fall outside admission. The original boundary reason stays in
the evidence, but it does not reopen the excluded record for infusion review.
Unidentified medications and source identity or detail-key conflicts remain
unresolved; excluded records do not change the infusion history.

### Check blank medication names with pharmacy

The reviewer approved `burden_v2.pharmacy_identity_lookup` on 2026-10-04.
Step 05 now reads the required `hosp/pharmacy.csv`, configured in
`configs/ehr.json`, and looks up only eMAR rows whose medication name is blank.
It requires matching nonempty `subject_id`, `hadm_id` and `pharmacy_id`, exactly
one pharmacy source row, and a nonempty pharmacy medication name. Duplicate rows,
multiple matches, absent matches, blank matched names and missing keys leave
identity unresolved. It never overwrites a nonempty eMAR medication name.

The original eMAR name stays unchanged. A separately recorded resolved name is
checked against the same confirmed exact-name medication dictionary. New names
remain unconfirmed until reviewed; no substring or route-based inference is used.
Pharmacy matches provide identity only: their order status, route and start/stop
times do not establish actual administration, IV infusion or a new infusion start.
Those decisions continue to use eMAR and its aggregated details. The official
[pharmacy documentation](https://mimic.mit.edu/docs/iv/modules/hosp/pharmacy.html)
describes the pharmacy-to-eMAR identifier link and the prescription fields.

After a run, inspect `outputs/audit/burden_v2/pharmacy_identity_audit.csv` for blank
eMAR names, lookup statuses, resolved names, dictionary confirmation and linked
raw pharmacy evidence. `pharmacy_identity_summary.csv` summarizes context rows,
deduplicated eMAR source records and patients. An event with a missing admission
ID may appear in multiple admission contexts; use the source-record counts when
counting independent eMAR records. A resolved identity does not mean the infusion
component or score is resolved. Review `dictionary_inventory.csv` for new pharmacy
names and the existing component review reports for remaining clinical conflicts.
Medication frequencies now include retained pharmacy source rows as well as eMAR;
source rows are not administration counts.

The lookup also preserves `medication_identity_audit.parquet` in the processed
`burden_v2` namespace. Event evidence includes the eMAR/detail keys and any cited
pharmacy keys, original and resolved names, match status, policy version and raw
pharmacy records. `run_metadata.json` records the lookup policy, summary and the
pharmacy input version. The shared Step 05 CSV tail check includes pharmacy.csv;
only a confirmed incomplete EOF record can be skipped. A missing pharmacy file,
missing required columns or an error in the middle stops the run and preserves
the previous published outputs.

Rerun only Step 05 after this update, using the usual command. No deletion of old
outputs or rerun of Steps 01-04 is needed when their source data is unchanged.

### Check remaining blank names with prescriptions

The reviewer approved `burden_v2.prescriptions_identity_lookup` on 2026-10-04.
Step 05 now also requires `hosp/prescriptions.csv`, configured in `configs/ehr.json`.
After pharmacy lookup, a still-blank identity can be supplemented from exactly
one prescription source row with the same nonempty `subject_id`, `hadm_id` and
`pharmacy_id` and a nonempty `drug`. Missing keys, absent matches, blank drug names,
duplicate rows and multiple matches remain unresolved. Multiple drug components
remain unresolved even if one row is marked `MAIN`; the program does not select
one component or combine names. Existing pharmacy conflicts remain unresolved.
Named eMAR records and resolved pharmacy names keep their existing identity.

The prescription provides medication identity only. Its route, order times and
drug type do not prove actual administration or a new infusion start. eMAR and
its details still govern those decisions. Resolved drug names use the existing
confirmed exact-name dictionary; new aliases need review. See the official
[prescriptions documentation](https://mimic.mit.edu/docs/iv/modules/hosp/prescriptions.html)
for the source fields and pharmacy identifier link.

On 2026-10-04, reviewer ZE Xiaokang confirmed 15 uniquely linked prescription
names as non-target: the three sodium chloride names, two 5% dextrose names,
Lactated Ringers, D5 1/2NS, D5NS, D5LR and six potassium-containing fluid names.
Their exact-name mappings are recorded in `configs/cohort.json` under dictionary
version `burden_partial_review_2026-10-04_v34`. Each mapping excludes only the
corresponding medication record and preserves any ongoing target infusion state.
The same reviewer also confirmed six exact names as non-target: `Amino Acids
5%-Dextrose 15%`, `Multivitamin IV`, `Trace Elements`, `Prismasate (B22 K4)`,
`Prismasate (B32 K2)` and `Sodium Chloride 3% (Hypertonic)`. These mappings also
exclude only the corresponding record and preserve target infusion history.
The current event rules are `burden_v2_audit_v17`. `Bag`, `SW` and blank names
have not been assigned a drug identity. Rerun only Step 05 to apply this update;
old results need not be deleted.

Inspect `outputs/audit/burden_v2/prescriptions_identity_audit.csv` for blank eMAR
records requiring this lookup, including unsuccessful or blocked lookups.
`prescriptions_identity_summary.csv` summarizes all originally blank eMAR records
by both lookup statuses, identity source and dictionary confirmation, with context,
deduplicated eMAR record and patient counts. `not_needed` means pharmacy already
supplied a name. The existing pharmacy report and `medication_identity_audit.parquet`
also include the prescription lookup status, version and match count. Raw evidence
preserves prescription records and their stable source keys alongside the original
eMAR name and any pharmacy evidence. Resolving a name does not guarantee a complete
clinical label. Unresolved identity remains visible in these reports even when
the study exclusion policy below removes the affected study from scoring.

The CSV tail report and run metadata include prescriptions.csv. Only a confirmed
incomplete EOF record may be skipped; a missing file, required-column failure or
middle parsing error preserves the preceding published outputs. The new reports
are published with the Step 05 transaction and included in its manifest hashes.

Run `python scripts/05_audit_burden_labels.py` after this update. You do not need to
delete old results or rerun Steps 01-04 when their inputs are unchanged. The run
replaces Step 05 outputs on success. Review the new reports before making further
clinical decisions; neither the horizon nor formal labels are frozen here.

### Exclude studies affected by unresolved medication identity

On 2026-10-04, reviewer ZE Xiaokang approved
`burden_v2.candidate_medication_identity_policy`. After the existing identity
lookups, Step 05 excludes an otherwise eligible study if its infusion evidence
still has an unresolved `unconfirmed_medication_name` decision in either audit
window. The union of affected study IDs is excluded in both the 48-hour and
72-hour outputs. The existing evidence rules include relevant history before
the image time, which is needed to distinguish continuation from a new start.
Records from other admissions or beyond the observation cutoff do not add an
exclusion. Multiple prescription matches remain ambiguous; the program does
not select a `MAIN` row or infer the identities of `Bag`, `SW` or a blank name.

Confirmed `Not Given` and `Not Started` records, and other confirmed record
exclusions, do not require a medication identity to establish non-administration.
They do not start or stop target infusion support. Missing identity alone does
not make these records unresolved; other clinical conflicts retain their checks.
Known non-target medications also exclude only the corresponding record.

Reviewer ZE Xiaokang confirmed a method-review precedence exception on
2026-10-04 for exact `Not Given` and `Not Started` states. When the only nonblank
administration type is one confirmed `action=review` value, that type alone no
longer makes the record unresolved. This includes `Standard Maintenance
Medication` in eMAR records `10005024-323` and `10030753-960`, affecting three
retained studies in run `f77fa52491204e05819902e5ebed6540`. Missing medication
names remain missing, and these excluded records cannot start or stop support.

The exception does not apply to missing types, unconfirmed types or multiple
distinct types. Positive `dose_given` contradicts non-administration; invalid,
negative or nonfinite doses remain unresolved. Missing doses or numeric zero
do not, by themselves, contradict these states. Existing ownership, temporal,
given-status, route and infusion-status checks remain in place. Other valid
administrations are assessed independently, and other clinical conflicts still
prevent scoring. The approved rationales are recorded in the two status mappings
under dictionary version `burden_partial_review_2026-10-04_v34`; the event rules
were `burden_v2_audit_v15` for that priority update. That update did not change
study exclusion decisions.

Run `ce60e837381646b4b9ac47c2b7bf19c9` verified that priority update: 489 candidates,
with 475 complete and 14 unresolved at 48 hours, and 471 complete and 18 unresolved
at 72 hours. The subsequent reviewed decisions below address all 18 unresolved
studies across the two windows.

Study exclusion is a cohort decision, not confirmation that an unknown drug is
non-target or that no infusion occurred. Excluded studies remain in the candidate
and observation tables, with null component flags, first-event times and scores.
Their exclusion reason, reviewer and policy version appear in the candidate table
and `outputs/audit/burden_v2/excluded_studies.csv`.

Inspect `outputs/audit/burden_v2/medication_identity_exclusions.csv` for each
triggering study, window and eMAR record. It includes original and resolved names,
both lookup statuses and match counts, stable source keys, raw evidence and the
original unresolved decision. `candidate_evidence.parquet` preserves the cited
source records and the reviewed exclusion policy. Both window audit reports and
`run_metadata.json` summarize excluded studies, admissions, affected patients and
distinct eMAR source records. Counts are deduplicated across windows; an affected
patient may still have other eligible studies. Review this sample loss before
comparing outcomes or freezing a cohort.

Only rerun `python scripts/05_audit_burden_labels.py`; Steps 01-04 need no rerun
when their inputs are unchanged. Do not delete old outputs, recovery logs or
backups. Successful publication replaces Step 05 outputs, including the exclusion
report and manifest hashes; failure preserves or restores the previous outputs.
Other clinical uncertainties still require review, and neither H nor formal
labels are frozen by this policy.

### Outputs and manual review

Intermediate tables live in `data/processed/cohort/burden_v2/`:

- `candidate_studies.parquet` and `candidate_evidence.parquet` preserve every
  unique-match study, eligibility decisions and source evidence.
- `window_48h/` and `window_72h/` each contain `candidate_observation.parquet`,
  `study_event_audit.parquet` and `event_evidence.parquet`, covering the same study
  keys. The evidence table is a long table with stable source hashes and raw values.
- Component detail tables preserve transfers, ventilation episodes, hypoxemia,
  aggregated medication administrations, infusion evidence and deaths.

Reports live in `outputs/audit/burden_v2/`. They include candidate counts,
conflicts, dictionary inventories, source coverage, component overlap, score
distributions, `window_comparison.csv`, `window_changes.csv` and deterministic
review samples. Each window also has `early_3h_events.csv` and
`early_discharge.csv`, including headers when empty. Counts distinguish studies,
admissions, patients and deduplicated source events. Early-discharge feature
comparisons currently use age and care level available at `t0`.

`care_interval_audit.csv` records each deduplicated transfer or ICU interval
once per candidate admission context, with raw times, parsed or inferred times,
the admission overlap, boundary flags, source keys and validation reasons.
`run_metadata.json` summarizes its row count and accepted boundary crossings.
An interval starting exactly at discharge can supply an event at that inclusive
window endpoint, while it has no duration inside the admission. No rounding
tolerance or time-zone conversion is applied.

Use a controlled CSV for manual event decisions. Set its location with
`burden_v2.overrides_path`; a relative path resolves from the project root.
Required columns are:

```text
sample_id,horizon_hours,component,event_flag,first_event_time,source_record_keys,reviewer,rationale,version
```

Each `(sample_id, horizon_hours, component)` may occur once. `horizon_hours` is
48 or 72; `component` is one of the six full column names. `event_flag` is 0 or 1.
A positive decision requires a reliable `first_event_time` inside that study's
window; a zero decision leaves that time empty. `source_record_keys` is a JSON
array of cited keys from the component evidence, quoted according to CSV rules.
`baseline_status` is an optional column with `not_observed`, `active_documented`
or `inactive_documented`. Cite the reviewer, rationale and review version.

`override_scope` is optional and defaults to `component`, retaining the previous
component-wide behavior. With `override_scope=source_record`, the decision must
have `event_flag=0`, an empty `first_event_time`, and no `baseline_status` override.
It excludes only the exact evidence rows listed in `source_record_keys` from that
sample, window and component. Citing a constituent of a merged episode is not
enough to exclude that episode. Cited rows must belong to the same patient and
admission. Other valid positives, first times and baseline evidence remain in
place; uncited conflicts still prevent scoring. A separate manual evidence row
records the review rationale without replacing the component result.

Rerunning applies overrides without overwriting the recorded automated decisions
or raw evidence. Event overrides cannot bypass candidate eligibility.

For an explicitly reviewed exclusion of one study, use the separate versioned CSV
at `burden_v2.candidate_exclusions_path`. Required columns are:

```text
sample_id,reason,source_record_keys,reviewer,rationale,version
```

Each sample may occur once. Cite a nonempty JSON array of source keys from that
patient's linked admission context, with a reason, reviewer and review version.
Medication citations may include linked eMAR, aggregated detail, pharmacy and
prescription source keys. Chart citations may include exact `chartevents` keys
with the same patient and explicit admission ID. The program checks each cited key against the current
admission context; unavailable keys and other-patient or other-admission records
stop the run before publication. Citing a record does not change its timestamp
or resolve its original clinical uncertainty.
These decisions only exclude the named study, in both windows; they cannot make
a candidate eligible or exclude other studies from the same patient or admission.
The excluded study's component flags and scores remain null. Candidate tables and
`excluded_studies.csv` show the reason and review provenance. `candidate_evidence.parquet`
preserves the original automated status and reasons alongside the manual decision
and cited source records. `run_metadata.json` records the exclusion CSV path, hash
and decision count; the shared CSV tail report includes this input as well.

On 2026-10-04, reviewer ZE Xiaokang approved exclusion of the 51 named studies in
the pre-admission medication batch from run `77473a7e6847489990daec850ee0404a`.
The previous file `configs/reviews/candidate_exclusions_2026-10-04_v4.csv`
retains the five earlier decisions and adds 51 study-specific decisions. The
new reason is `Pre-admission medication record with unresolved identity`. Each
decision cites the admission boundary and affected medication source records,
preserving the before-admission charttime and unresolved drug identity. This
approval applies to the named studies only, not every future pre-admission record.
Previous review CSV versions remain available.

On 2026-10-04, ZE Xiaokang approved 14 additional study exclusions from run
`ce60e837381646b4b9ac47c2b7bf19c9`: two for medication event uncertainty and 12 for
invasive-ventilation evidence or onset uncertainty. The configured file is now
`configs/reviews/candidate_exclusions_2026-10-04_v5.csv`, retaining all 56 earlier
decisions and adding these 14, for 70 explicit study decisions. Existing identity
and care-unit policies continue separately. An exclusion does not establish
that the clinical event was absent, and does not remove every study from an
affected patient.

Four other studies remain eligible. The configured
`configs/reviews/event_evidence_reviews_2026-10-04_v1.csv` contains seven
`source_record` decisions: both windows for studies 53858061, 51834590 and 55475917,
and only 72 hours for 54628956. They exclude five distinct reviewed chart records:
one BiPAP/CPAP record, three SpO2 records before ICU admission, and one SpO2 record
with warning=1. The direct invasive-ventilation positive for 53858061 and the valid
hypoxemia positive for 51834590 remain. No time is rounded or shifted, and these
decisions do not automatically apply to future records. See
`configs/reviews/step_05_review_decisions_2026-10-04_v5.md` for the approved list.

The event rules are `burden_v2_audit_v17`, and the dictionary remains
`burden_partial_review_2026-10-04_v34`. Review inputs have their own versions and
hashes. Rerun only Step 05, without deleting old outputs or rerunning Steps 01-04.
The current-output preview implies 475 eligible studies, 304 admissions and 222
patients, with 475 complete and zero unresolved studies in each window. These
counts must be verified by the next full run. Zero unresolved studies does not
freeze labels, confirm every dictionary inventory entry, or select H. Check
`excluded_studies.csv`, `candidate_evidence.parquet`, both windows' event evidence
and `run_metadata.json` after that run.

Candidate source conflicts remain pending unless resolved through source or mapping
review, or handled by an explicit study exclusion or a confirmed conflict policy.
ZE Xiaokang approved `candidate_conflict_policies.non_icu_transfer_with_active_icustay`
on 2026-10-04: exclude a study when its unique confirmed transfer location is a
ward or Stepdown and a valid icustay also covers `t0`. The reason is
`Conflicting care-unit sources at t0`. This policy does not resolve unknown or
simultaneous conflicting transfer locations, reverse-direction mismatches, or
conflicts occurring only after `t0`. Those retain the existing rules.
The policy is versioned and records its reviewer and rationale. Explicit study
exclusions take precedence; policy decisions use the same provenance columns and
candidate evidence, citing the configuration file and the active source intervals.
`run_metadata.json` records the explicit CSV count, additional policy exclusion
count and their total, plus the policy snapshot. Both windows exclude affected
studies with null flags and scores; other studies in the admission are evaluated
at their own `t0`.

Update source versions and rebuild dependent upstream steps if underlying data
changes. Rerun Step 05 after updating review files; existing published reports
retain their previous decisions until a successful replacement run.

Inspect `run_metadata.json` after every run:

- `audit_checks_passed` records structural and score checks.
- `dictionary_review_complete` means every observed value has a confirmed
  classification. It can remain false when clinical identity is unknown.
- `dictionary_handling_review_complete` means every observed value has either a
  confirmed classification or an explicitly approved uncertainty-handling policy.
- `audit_ready_for_window_decision` additionally requires eligible candidates and
  no unresolved candidate or event evidence. Handling approval cannot bypass
  these checks.
- `ready_for_formal_labels` and `horizon_frozen` remain false in Step 05. The group
  must choose H and freeze the reviewed rules before implementing Step 06.

On 2026-10-04, ZE Xiaokang approved the existing handling of exact care-unit
values `UNKNOWN` and `Unknown`, and normalized medication keys `""`, `bag` and `sw`.
Their identities remain unknown; their original classification confirmation
flags remain unchanged. The independent `burden_v2.dictionary_handling_reviews`
configuration records `handling=preserve_unknown`, `handling_confirmed`, reviewer,
date, rationale and version `dictionary_handling_review_2026-10-04_v1`.
It does not classify these drugs as non-target or these units as ordinary wards.
Existing identity lookups, non-administration rules, study exclusions and clinical
conflict rules continue to determine each record and sample. An unapproved new
value or a new clinical conflict still prevents audit readiness.

`dictionary_inventory.csv` preserves `confirmed` and adds `handling_confirmed`,
`handling_policy`, `handling_reviewer`, `handling_reviewed_on`,
`handling_rationale` and `handling_version`. The window audit reports explain the
distinction. Metadata includes both completion flags, the handling-review
configuration and its SHA-256. Existing configurations without this optional
section retain classification-based readiness. The dictionary classification
version remains `burden_partial_review_2026-10-04_v34`.

ZE Xiaokang selected **72 hours** for the primary question: deterioration burden
within three days after imaging. The versioned selection record is
`configs/reviews/window_selection_2026-10-04_v1.md`. Step 05 still audits both
windows. Run `e80f9e9d77624aa2bd0be44a7e24c3cc` verified handling readiness, 475
eligible studies, 304 admissions and 222 patients, with zero unresolved labels in
both windows. The reviewed primary horizon and label-rule versions are now frozen
in `outputs/audit/burden_v2/window_decision.md` and
`outputs/audit/burden_v2/endpoint_decision.json`. The JSON binds the supporting
document, verified audit, configurations, review inputs and selected-window
artifacts through SHA-256 hashes. Its publication follows the supporting document.
Consumers must verify those hashes before using the decision. Do not switch the
primary window using subsequent model performance.

No additional Step 05 rerun is needed for this decision. Step 05 metadata remains
an immutable snapshot with `horizon_frozen=false` and
`ready_for_formal_labels=false`; the subsequent manual decision records the freeze
separately. Formal labels and the common training cohort remain unpublished.
Step 06 can now build formal labels. An upstream change or a later audit replacing the cited manifest
requires revalidation and a new reviewed decision version; do not silently reuse
the old frozen JSON or edit prior audit metadata.

Step 05 uses the existing `outputs/audit/step_05_manifest.json` and recovery
protocol. Missing columns, source changes, invalid overrides or publication
errors preserve the previous output set. A complete manifest means the audit
files were published together; it does not authorize formal label release.

DuckDB scans and filters CSVs into a temporary disk database, with its memory
limit controlled by `configs/ehr.json`. Python processes one candidate admission's
filtered history at a time. DuckDB's limit does not cap Python memory; leave room
for a large admission and for staging, spill files and publication backups.
Temporary data is removed on ordinary completion or failure. Raw tables,
patient-level outputs and review overrides remain in controlled storage.

## Step 06: formal burden labels

After the reviewed endpoint decision is frozen, run only:

```powershell
python scripts/06_build_burden_labels.py
```

The script uses the same optional `--paths`, `--cohort`, `--ehr` and `--imaging`
configuration arguments as Step 05. The horizon comes exclusively from
`outputs/audit/burden_v2/endpoint_decision.json`; there is no CLI horizon override.
The current approved horizon is 72 hours. No Step 01-05 rerun or deletion is
needed while the frozen inputs remain unchanged.

Step 06 verifies Steps 03-05 manifests, the frozen decision and supporting
documents, configuration and review hashes, audit code, upstream images/matches,
and raw CSV size/modification-time versions. It does not scan raw CSVs or decode
DICOMs. It independently reconstructs all six flags and first times from final
Step 05 evidence, verifies ownership and observation boundaries, and recalculates
the capped score and derived endpoints. The reconstructed results and counts must
match the frozen audit. Eligible unresolved studies, missing fields, duplicate
keys, conflicting evidence, stale sources or unfrozen decisions stop publication.
Unknown clinical identities may remain unknown after their handling was reviewed;
`dictionary_review_complete=false` alone does not block this step.

Published products are:

- `data/processed/cohort/burden_v2/study_burden_labels.parquet`: completed eligible
  labels, image/study keys, boundaries, six flags and first times, capped domains,
  scores, coverage and version provenance. IDs are int64, flags uint8, scores and
  observation hours float32, and timestamps timezone-free microseconds.
- `data/processed/cohort/burden_v2/event_evidence.parquet`: final 72-hour evidence
  for published studies, including individually excluded records, original
  fields, automated decisions and manual review. Other studies' evidence and
  study exclusions remain in Step 05.
- `outputs/audit/burden_v2/early_3h_events.csv` and `early_discharge.csv`: formal
  reports with headers even when empty. Early events count toward the main score;
  events before alive early discharge remain in the label.
- `outputs/audit/burden_v2/step_06_report.md` and `step_06_run_metadata.json`:
  counts, component/score summaries, early events/discharge, coverage and frozen
  source/version hashes.
- `outputs/audit/step_06_csv_tail_checks.csv` and `step_06_manifest.json`: this
  step's `no_csv_inputs` report and transaction completion/checksums.

For the current frozen audit, expect 475 studies, 304 admissions and 222 patients,
with 34 nonzero scores, 16 `phys_death` positives and 12 `high_burden` positives.
Inspect `label_checks_passed=true` and `formal_labels_published=true` in Step 06
metadata and verify its manifest. A successful rerun replaces only Step 06
products; a failed run preserves or restores the previous set. Do not manually
delete recovery journals or backups.

Step 05 reports and the frozen decision remain immutable historical snapshots.
Their earlier `ready_for_formal_labels=false` or `formal_labels_published=false`
fields describe their creation time; the Step 06 manifest and metadata record
the later formal publication. The two accepted incomplete-EOF source warnings
remain visible through the Step 05 tail report. No recorded event is an
operational zero, not proof that the clinical event never occurred.

Formal labels are not the common training cohort. Step 06 produces neither
`cohort.parquet` nor `cohort_audit.parquet`, splits, EHR features or a training
package. Step 07 image QC and Step 08 common-cohort freezing follow separately.

Run all synthetic checks without scanning the real datasets:

```powershell
python -B -m unittest discover -s tests/cohort -p "test_*.py" -v
ruff check --no-cache src/cohort src/imaging scripts tests/cohort
```

These checks exercise both windows, clinical boundaries, support continuity,
source ownership, overrides and failed publication. Full-data runtime and actual
clinical dictionary review remain part of the user-run audit.

## Step 07: deterministic image preparation and QC

After Step 06 succeeds, run:

```powershell
python scripts/07_qc_and_prepare_images.py
```

No deletion or Step 01-06 rerun is needed while their sources remain unchanged.
Step 07 processes only the selected images for formal labels: currently 475
studies, 304 admissions and 222 patients, with 317 AP and 158 PA images. It
verifies Steps 02-06 manifests, the frozen 72-hour decision, label provenance,
image keys, paths, view, acquisition time and indexed dimensions. It does not
select another image or change any outcome label. Source metadata contradictions
stop publication; missing or undecodable individual files receive QC failures
while other images continue.

The four common configuration arguments remain optional. `--processing` defaults
to `configs/image_processing_v1.json`; keep image-processing settings independent
of the frozen `imaging.json` and endpoint configurations. The V1 settings specify
224 x 224, valid-pixel 1st/99th percentiles using NumPy's linear method, bilinear
resize, nearest-neighbor masks, half-up dimension rounding and zero padding.
Review sampling uses seed 42 and at most five normal images per
view/polarity/modality/VOI/presentation processing group.

Processing revision `burden_v2_images_v2` corrects the DX VOI LUT depth check:
Digital X-Ray for Presentation/Processing permits 10-16 bits under
[DICOM PS3.3 C.8.11.3.1.5](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.11.3.html#sect_C.8.11.3.1.5).
General Modality/VOI LUT depth checks remain 8/16 bits. The first Step 07 run
(`23c130f4096e4d2aab5670d8da9757c9`) incorrectly flagged DX 14-bit VOI LUTs and
used windows instead. Rerun only Step 07 before reviewing its new previews and
fixed sample list. Header-only inspection identified 117 affected DX files with
valid 14-bit VOI LUTs. Two other files have CR SOP classes, 12-bit VOI LUTs and
no window; their metadata uncertainty remains pending review under the general
VOI rule. The configuration filename stays `image_processing_v1.json`;
its independent processing revision and SHA-256 change. Previous source/config/NPY
review fingerprints cannot authorize the revised outputs. Steps 05-06 and the
frozen endpoint decision remain unchanged.

The shared transformation decodes a single grayscale frame, identifies padding
in raw pixel space, and applies a valid Modality LUT or rescale (defaults 1/0),
the first valid VOI LUT or window, then one display-direction conversion.
Presentation LUT output is treated as P-values; shape and photometric metadata
must agree. Without a presentation transform, MONOCHROME1 is inverted once and
MONOCHROME2 is retained. Non-padding pixels determine percentile clipping to
[0,1]. Aspect ratio is preserved when resizing; odd leftover borders go below
or to the right. Padding remains zero. Each NPY is reloaded with
`allow_pickle=False` and checked for float32 type, 224 x 224 shape, finite values,
range and SHA-256. There is no augmentation, crop, rotation, text removal,
three-channel replication or model-specific normalization.

Published products are:

- `data/processed/imaging/burden_v2/image_processing.parquet` and
  `image_qc_status.parquet`: one row per formal-label study, including failures,
  with source versions/hashes, transformations, dimensions, percentiles,
  padding, current NPY reference/hash, automated status and manual decisions.
- `data/processed/imaging/burden_v2/images/pXX/p<subject_id>/s<study_id>.npy`:
  completed deterministic transformations only. Failed or unreliable
  transformations have no NPY; zero-image substitutes are never produced.
- `outputs/qc/burden_v2/image_qc_review.html` and its PNG previews: local review
  page showing before/after images, display method, flags and errors. Outcome
  scores and event labels are hidden. An available raw-value preview for an
  incomplete transform is explicitly marked as non-diagnostic.
- `outputs/qc/burden_v2/review_sample.csv`, `duplicate_images.csv`,
  `dicom_failures.csv` and `image_qc_review.csv`: fixed normal-image sample,
  source/decoded-pixel/NPY exact duplicate groups, processing failures and the
  required-review template. Empty reports retain headers.
- `outputs/qc/burden_v2/qc_report.md`, `run_metadata.json`, and
  `outputs/audit/step_07_csv_tail_checks.csv` / `step_07_manifest.json`:
  counts, review progress, provenance, CSV checks and publication checksums.

Normal images passing automatic checks and outside the fixed sample are `pass`.
Sampled images, metadata concerns and suspected duplicates are `pending_review`;
duplicates are never automatically discarded. Hard failures are `fail`, with
their manual acknowledgment tracked separately. `image_checks_passed=true`
means the run's structural checks passed, not that every image passed QC.
All required reviews must be resolved before `qc_review_complete=true`.

After the first run:

1. Open `outputs/qc/burden_v2/image_qc_review.html` in a browser. It initially
   shows required reviews; the button also exposes all other images. Inspect
   grayscale direction, lung coverage, proportions, padding, clipping, artifacts,
   text and AP/PA. Review every failure, anomaly, duplicate and sampled image.
2. Copy the generated `image_qc_review.csv` to an independent versioned input,
   for example `configs/reviews/image_qc_reviews_v1.csv`. Keep `sample_id` and
   all three fingerprints unchanged. Fill `decision` (`pass` or `exclude`),
   `rationale`, `reviewer`, ISO `reviewed_on` date and a single `version` for the
   file. Preserve the optional `source_run_id` provenance column. For partial
   review batches, include only completed decisions; unreviewed studies remain
   pending. Do not fill in a missing source or NPY hash.
3. Rerun only Step 07 with the versioned decisions:

   ```powershell
   python scripts/07_qc_and_prepare_images.py --reviews configs/reviews/image_qc_reviews_v1.csv
   ```

A manual pass requires a completed, validated NPY and cannot bypass a hard
failure. Such an image needs corrected processing and a new run, or an explicit
exclusion. A decision applies only to the exact source DICOM, processing
configuration and NPY hashes; changed fingerprints stop publication rather than
silently reusing old approval. Unknown sample references, conflicting decisions
and invalid CSV fields also stop publication. An incomplete final review record
is warned about and skipped under the shared EOF policy; earlier valid decisions
still apply and the omitted study remains unreviewed. Review the file's source
manually if its tail check reports a warning.

A successful rerun transactionally replaces current Step 07 files; failure
preserves or restores the previous set. Do not delete recovery journals or
backups. Only files referenced by current tables and the current manifest are
current outputs: older unreferenced NPYs or previews may remain on disk. Step 07
does not create the final image manifest, common cohort, patient split, features
or training package. Once QC and required reviews pass, Step 08 can separately
freeze the common cohort.

## Step 08: freeze the common cohort

After Step 07 reports `image_checks_passed=true`, `qc_review_complete=true` and
`ready_for_cohort_freeze=true`, run:

```powershell
python scripts/08_freeze_common_cohort.py
```

No deletion or rerun of Steps 01-07 is needed. Step 06 contains all completed
formal labels; Step 08 freezes their intersection with images that passed QC.
The expected `burden_v2_h72_r01` release has 473 studies, 303 admissions and 222
patients, including 34 nonzero scores, 16 `phys_death` and 12 `high_burden`.
The audit retains all 1,537 candidates: 1,062 upstream exclusions, two image
exclusions and 473 inclusions. QC exclusions 52318649 and 57900053 remove one
admission but do not remove an entire patient; that patient has other included
studies. Actual counts must reconcile with the verified upstream artifacts.

The four common configuration arguments remain optional. `--processing`
defaults to `configs/image_processing_v1.json`. The horizon comes exclusively
from the frozen endpoint decision; there is no horizon or review override.
The human-review input is located through Step 07 metadata and hashed without
parsing CSV. Steps 02-07 manifests, current configuration/code content hashes,
upstream run IDs, review hashes and original file versions must agree. A later
Git commit is allowed when the bound file contents remain unchanged. Clinical
CSV records are not rescanned; DICOM pixels are not decoded or processed.

Step 08 independently reconstructs events, first times and capped scores from
the published final evidence, validates ownership and observation boundaries,
and compares every formal field with the audited labels. Every formal-label
sample must have exactly one resolved QC row. Included NPYs are reloaded with
`allow_pickle=False` and checked for protected paths, hashes, float32 type,
224 x 224 shape, finite values and [0,1] range. Any mismatch stops publication;
it does not create new exclusions, fill missing labels or change the decision.

Current products are:

- `data/processed/cohort/burden_v2/cohort.parquet`: the shared analysis cohort,
  retaining all formal-label fields and types, with NPY paths/hashes, source
  DICOM hashes, processing and human-QC provenance, and `cohort_version`.
  NPY paths are relative to `data/processed/imaging/burden_v2/` and use `/`.
- `data/processed/cohort/burden_v2/cohort_audit.parquet`: all Step 05 candidates,
  with inclusion flags, exclusion stage/reason and QC provenance. Candidates
  never sent to image QC have `qc_performed=0` and null QC fields. Upstream
  excluded labels remain null; image-excluded studies retain their formal
  labels and are not converted to zero scores.
- `outputs/audit/burden_v2/cohort_flow.csv`, `cohort_qc_exclusions.csv`,
  `training_feasibility.csv` and `source_overlap.csv`: study/admission/patient
  counts, human exclusion reasons, all score bins 0-8, component/derived
  endpoints, coverage, early events/discharge, study counts per patient and
  stable source associations. Source overlap includes baseline and excluded
  evidence, with qualifying associations counted separately. A source record
  is deduplicated by component and stable key; it is not an independent patient.
- `outputs/audit/burden_v2/cohort_freeze.json`, `step_08_report.md` and
  `step_08_run_metadata.json`: frozen sources, release, counts, review and
  output hashes, plus original freeze and latest verification run IDs.
- `outputs/audit/step_08_csv_tail_checks.csv` and `step_08_manifest.json`:
  `no_csv_inputs` and transactional publication hashes. The report references
  the accepted Step 05 source warnings and Step 07 review CSV checks. Tail
  warnings do not establish that clinical records are complete.

The first successful publication also saves tables and reports under
`data/processed/cohort/burden_v2/releases/burden_v2_h72_r01/` and
`outputs/audit/burden_v2/releases/burden_v2_h72_r01/`. A repeat with the same
version and source/configuration/code fingerprints verifies and reuses these
snapshots, preserving their bytes and hashes. Current-path products are
published as one transaction; failure or interruption restores the previous
set. Do not manually remove recovery journals or backups. Step 05-07 metadata,
formal labels and the frozen endpoint decision remain unchanged.

An already frozen version cannot accept changed sources, code, configuration
or samples. After reviewing a deliberate new release, specify its version:

```powershell
python scripts/08_freeze_common_cohort.py --release-version burden_v2_h72_r02
```

The version must match the frozen horizon. A new revision preserves older
snapshots and switches the current products together. Snapshots retain tables,
reports and hash provenance; they depend on the image cache and do not copy
NPYs or constitute portable training packages. Early three-hour events and
events before alive early discharge remain in the original labels. All later
models must use the same frozen common cohort. Patient splitting (Step 09),
EHR features, final image manifests and training packages are subsequent steps.
