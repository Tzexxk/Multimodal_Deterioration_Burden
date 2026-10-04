# Laboratory candidate selection

ZE Xiaokang approved this selection on 2026-10-05. The scope includes all 368
observed itemids. The expected result is 25 selected and 343 deferred itemids.
This records candidate scope. Clinical dictionary approval and EHR feature
readiness remain pending.

## Reproduce using verified caches

From the repository root in the ICU_Risk environment:

```powershell
python -m src.ehr.lab_selection `
  --policy configs/reviews/lab_candidate_policy_2026-10-05_v1.json `
  --output configs/reviews/lab_candidate_selection_2026-10-05_v1.csv
```

The command reads cached laboratory evidence and frozen upstream artifacts;
it never scans the raw laboratory CSV or republishes Step 10 outputs. It checks
audit hashes, source/configuration/code bindings, and frozen cohort/split
identities. Identical selection CSV/JSON pairs are reused. Changed content
requires a new output version. The JSON sidecar binds the policy, audit,
training sample keys, patient split, selection code and CSV hashes.

The CSV contains one row for every observed itemid, including pending and
deferred candidates. It contains aggregate coverage only, without patient IDs,
sample IDs, split assignments or outcomes. Do not pass this CSV to --reviews;
that interface accepts completed clinical dictionary decisions.

## Coverage and decisions

Only the 337 training studies and 156 training patients determine selection.
Each study counts once per itemid when at least one finite valuenum result is
available by t0, belongs to the study admission, and has charttime in the
inclusive [t0-24h, t0] window. Units are inventoried separately. Counting an
item's coverage across units does not approve a conversion or combine values.
Coverage precedes final specimen, numeric, duplicate and validity rules.

Use exact fractions: missingness <30% selects automatically; 30%-50% and
>50%-80% require an explicit individual decision; >80% defers automatically.
Missingness exactly 30%, 50% or 80% requires an individual decision. Missing
decisions remain pending. CSV decimal fractions are descriptive; integer
missing/total counts are the exact values used for threshold comparisons.

The policy file records the three approved 30%-50% items and seven decisions
in the >50%-80% band. After clinical cleaning, reassess items crossing a band;
PT and alkaline phosphatase remain explicitly deferred until a new decision.
Use a new policy/output version for any reassessment. Validation and test data
do not choose candidate scope or preprocessing parameters.

## Pending dictionary work

MCHC (51249) has percent and g/dL source branches. INR (51237) has a missing
source unit. Resolve these branches before dictionary freezing. Every retained
item still needs supported specimen, duplicate and numeric selectors, canonical
units, finite conversion coefficients and explicit validity-range treatment.
Sparse bilirubin, ALT, AST, INR and PTT candidates require training-only
stability assessment; external evidence does not establish local predictive
benefit. ALT/AST and PT/INR overlap has not been empirically assessed here.

The 25 itemids do not specify which aggregation columns will enter the final
feature whitelist. Preserve all 473 sample keys, the 72-hour outcome, the frozen
patient split and upstream snapshots. Aggregation and Steps 11-12 follow
dictionary review. No clinical rules or scope acknowledgment are signed by
this selection record.

## Background evidence

Total bilirubin and INR appeared in a gastroenterology-ward ICU-transfer
prediction study with a different population and outcome definition:
[Kim et al., 2017](https://pubmed.ncbi.nlm.nih.gov/28445859/).
For measurement interpretation, see NIH MedlinePlus on
[PT/INR](https://medlineplus.gov/lab-tests/prothrombin-time-test-and-inr-ptinr/),
[PTT](https://medlineplus.gov/lab-tests/partial-thromboplastin-time-ptt-test/),
[ALT](https://medlineplus.gov/lab-tests/alt-blood-test/),
[AST](https://medlineplus.gov/lab-tests/ast-test/) and
[alkaline phosphatase](https://medlineplus.gov/lab-tests/alkaline-phosphatase/).
