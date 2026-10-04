# Primary observation window selection

- Review version: `window_selection_2026-10-04_v1`
- Reviewer: **ZE Xiaokang**
- Decision date: **2026-10-04**
- Selected primary horizon: **72 hours**
- Status: reviewed selection, pending verification of the updated Step 05 audit
- Source audit: `d0eef2c05e3b4e6ab9e41e3a5b03f893`
- Source rules: `burden_v2_audit_v16`
- Updated audit rules: `burden_v2_audit_v17`
- Classification dictionary: `burden_partial_review_2026-10-04_v34`

## Research question and rationale

The primary research question is deterioration burden within three days after
imaging. ZE Xiaokang explicitly selected the 72-hour option after reviewing the
48-hour and 72-hour audit comparison. Event counts describe the two observation
windows; they are not an automatic rule for choosing H. The primary window must
not be switched using subsequent model performance.

The event window remains `(t0, min(t0 + 72 hours, dischtime)]`. The outcome remains
`max(ICU, Stepdown) + 2 * max(IMV, hypoxemia) + 2 * infusion + 3 * death`, with a
range of 0 to 8. Early events and events before an early discharge remain included
under the existing V2 rules.

## Source comparison

| Measure | 48 hours | 72 hours |
| --- | ---: | ---: |
| Eligible studies | 475 | 475 |
| Eligible admissions | 304 | 304 |
| Eligible patients | 222 | 222 |
| Studies with burden score greater than zero | 25 | 34 |
| Patients with a nonzero study score | 20 | 24 |
| Studies with a physiological event or death | 11 | 16 |
| High-burden studies | 10 | 12 |
| Unresolved eligible studies | 0 | 0 |

## Pending audit verification and formal freeze

The updated audit distinguishes unknown clinical identity from approved handling
of that uncertainty. `UNKNOWN`, `Unknown`, blank medication names, `Bag` and `SW`
remain unidentified. Their approved handling does not alter clinical evidence or
force unresolved labels to zero.

Rerun Step 05 and verify structural checks, handling review completion, audit
readiness, clinical counts and preserved positive evidence. Both horizons remain
in the audit. This document does not authorize formal label publication, set
`horizon_frozen`, or create an already-frozen `endpoint_decision.json`.

After verification, formal window decision files must cite the new verified run
and its actual rule configuration hash. The prior audit hash must not be used to
freeze the updated rules. Step 06 is outside this implementation round.
