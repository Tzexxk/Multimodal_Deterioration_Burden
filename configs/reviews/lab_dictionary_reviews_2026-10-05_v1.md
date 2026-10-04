# Approved laboratory dictionary inputs

ZE Xiaokang approved MCHC, INR, the common dictionary rules and the complete
version-specific scope on 2026-10-05. The clinical review CSV contains 370
explicit itemid/unit decisions: 26 retained branches for 25 itemids and 344
excluded branches for 343 itemids. Every observed branch remains traceable.

The separate scope CSV binds the complete original scope inventory hash
`5c52abb8bf118b7803ebb5bee7e3c2f5db3575454dba7fc8a2e69f0876ccc4ef`.
Exclusion applies to this dictionary version. Future reconsideration requires
new reviewed inputs and a new dictionary version.

The input manifest binds these files to the existing selection, audit, sources,
frozen cohort/split identities and input-builder code. Its read-only validation
passed all 370 branches and 26 retained conversion checks. Production review
and dictionary freeze have not been published. Historical candidate-selection
files retain their original pre-approval status.

## Approved clinical rules

Retained itemids use their explicitly approved Blood dictionary identity.
Source units remain exact. All conversions have multiplier 1 and offset 0.
MCHC percent and g/dL branches both map to mchc in g/dL. INR's empty source
unit maps to inr with canonical unit 1 and missing_unit_rule explicit_item_unit.
These approvals are item-specific and do not authorize other unit inferences.

MCHC scale evidence came from full cached patient history, separately from
training-study coverage. Among unambiguous same-patient/admission/specimen/time
Hb/Hct/MCHC pairs, 6,760 of 6,770 percent pairs and all 5,357 g/dL pairs had
absolute difference at most 0.5 from 100*Hb/Hct. This is descriptive evidence
for the approved reporting convention. The remaining ten discrepancies stay
in the bound source cache and retain traceability; no consistency-based
exclusion, numerical replacement or recalculation is approved.

Use only finite valuenum results. Missing/non-numeric and comparator results
remain missing without guessed thresholds. Numeric/text disagreement blocks
acceptance. Keep the latest t0-available same-specimen/time revision under the
existing deterministic policy; conflicting IDs and missing specimen IDs block
acceptance. Future results cannot change eligibility at an earlier t0.

Validity bounds remain empty with an explicit basis: no reliable hard bounds
were approved for this version. Reference ranges describe source evidence;
finite extreme values remain intact. No arterial/venous mapping is introduced.

## Run production review

From the repository root:

```powershell
conda activate ICU_Risk
python scripts/10_build_ehr_features.py --mode review `
  --reviews configs/reviews/lab_dictionary_reviews_2026-10-05_v1.csv `
            configs/reviews/lab_dictionary_scope_2026-10-05_v1.csv
```

Inspect review_state.json, review_progress.csv and lab_conversion_checks.csv
under outputs/audit/burden_v2/labs/lab_audit_v1/. The production run must report
review_complete and scope_acknowledgment_complete true, zero pending branches,
and no unresolved retained-rule conflicts. Codex verifies these artifacts
read-only before the user runs freeze.

## Freeze after successful review verification

```powershell
python scripts/10_build_ehr_features.py --mode freeze `
  --reviews configs/reviews/lab_dictionary_reviews_2026-10-05_v1.csv `
            configs/reviews/lab_dictionary_scope_2026-10-05_v1.csv `
  --dictionary-version lab_dictionary_v1
```

Existing publication recovery and immutable-version checks remain in force.
The dictionary includes both retained and excluded rules. Verify 25 retained
itemids, 26 retained branches, current/snapshot hashes and the freeze record.
Dictionary freezing leaves ehr_features_ready false.

## Subsequent work

Decide aggregation columns, static features and the final feature whitelist
separately. Recompute clinically cleaned training missingness and reassess
items crossing selection bands; PT and ALP retain their explicit version-specific
exclusion. Sparse features require patient-grouped training-only evaluation.
Do not use validation/test distributions to select feature scope.

The 72-hour outcome, 473 sample keys, frozen patient assignments and upstream
snapshots remain unchanged. The user runs production commands. Codex runs
synthetic tests and read-only evidence checks. Steps 11-12 follow accepted EHR
features; no commit, push or dependency addition is required.

INR's ratio interpretation is documented by
[NIH MedlinePlus](https://medlineplus.gov/lab-tests/prothrombin-time-test-and-inr-ptinr/).
