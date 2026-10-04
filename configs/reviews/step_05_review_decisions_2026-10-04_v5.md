# Step 05 reviewed decisions, 2026-10-04

Reviewer: **ZE Xiaokang**. Source run: `ce60e837381646b4b9ac47c2b7bf19c9`.
Rules after implementation: `burden_v2_audit_v16`. Dictionary:
`burden_partial_review_2026-10-04_v34` (unchanged).

## Study exclusions

`candidate_exclusions_2026-10-04_v5.csv` retains all 56 previous decisions and adds
14 named studies. Both observation windows exclude each named study; all six
component flags, component first times and derived scores remain null.

| Reason | Study IDs |
| --- | --- |
| Medication event cannot establish actual administration or new onset | 57156853, 52610464 |
| Invasive ventilation evidence or onset cannot be verified | 50395086, 58640215, 57567183, 50551427, 53441481, 50651554, 52621553, 57295622, 51777274, 56720087, 59258773, 56562738 |

Each CSV row contains the full sample ID, cited source keys, reviewer, review
version, source run, study time and specific unresolved records. Documented
positives remain part of the source history; exclusion does not mean an event
was absent. Other studies from the same patient or admission remain independent.

## Source-record exclusions with retained studies

`event_evidence_reviews_2026-10-04_v1.csv` contains seven `source_record` decisions
covering five distinct chart records. Event flag 0 applies to those cited records,
not to the entire component. The existing component and scoring functions assess
all remaining evidence. Raw evidence and automated decisions remain traceable.

| Study | Windows | Approved record treatment | Remaining evidence |
| --- | --- | --- | --- |
| 53858061 | 48, 72 hours | Exclude the BiPAP/CPAP chart record at 2130-09-19 14:40:00 from new invasive-ventilation evidence. | Keep the direct invasive-ventilation positive, first time 2130-09-18 16:15:00. |
| 51834590 | 48, 72 hours | Exclude SpO2 94% at 2170-07-28 23:20:00, 50 seconds before ICU admission, and SpO2 90% with warning=1 at 2170-07-30 00:00:00. | Keep valid severe-hypoxemia evidence, first time 2170-07-29 20:30:00. |
| 54628956 | 72 hours | Exclude SpO2 88% at 2153-03-11 01:50:00, 17 seconds before ICU admission. | Assess other valid records; the current run has no qualifying hypoxemia positive. The 48-hour result is unchanged. |
| 55475917 | 48, 72 hours | Exclude SpO2 96% at 2126-05-22 19:50:00, 57 seconds before ICU admission. | Assess other valid records; the current run has no qualifying hypoxemia positive. |

No timestamp is changed. Only named samples, components, windows and exact source
keys are covered. An uncited new conflict remains unresolved; these decisions are
not general rules for noninvasive ventilation, warning records or ICU boundaries.

## Expected reconciliation and next run

The published source run has 489 eligible studies. Removing 14 yields a preview
of **475 eligible studies, 304 admissions and 222 patients**, with 475 completed
labels and zero unresolved studies in each window after the record reviews.
Actual counts must be verified after rerunning Step 05.

Run `python scripts/05_audit_burden_labels.py` only. Do not delete previous outputs:
successful publication replaces them, while failure retains or restores them.
Check the exclusion report, candidate evidence, window evidence, CSV tail checks
and run metadata. This round does not select H, freeze labels, confirm every
dictionary inventory entry, or create training samples.
