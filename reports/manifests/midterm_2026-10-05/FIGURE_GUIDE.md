# Midterm figure captions and reproduction

Run from the repository root: `python scripts/generate_midterm_report.py`.
Figures are English PNG (220 dpi) and editable-text SVG; tables are aggregate CSV.
The script reads processed artifacts only. It does not re-run cleaning or train models.

1. **01_cohort_flow:** The available local images yield a shared 473-study cohort.
   Images and studies are different units. Exclusion reasons are in the companion CSV.
   Patients at excluded stages overlap patients retained elsewhere; do not subtract patient counts.
2. **02_time_windows:** Information must be available by the CXR time. This is a method
   schematic, not an individual timeline; future events contribute only to labels.
3. **03_labels:** Final-cohort scores and overlapping components. Studies and unique
   positive patients are separate counts. A zero is absence of qualifying recorded evidence,
   not proof of no clinical deterioration. The defined 0-8 range includes unobserved classes.
4. **04_lab_missingness:** All retained laboratory features, denominator 473 studies.
   A value is present when last_24h is nonmissing after approved rules. Companion CSV includes
   train/validation/test coverage. Colors describe historical coverage bands, not new selection.
   Missing count_24h is represented by zero; value aggregations remain NaN.
5. **05_cleaning:** Joint availability/ownership rejection categories are mutually exclusive
   at the item-window association level. The left panel covers all observed items; the right
   covers reviewed feature processing. Do not add these stages into one total. Source counts
   in the per-item/per-feature CSV are distinct within each group, not a global unique total.
   This release has no finite value removed by an unreviewed clinical range rule.
6. **06_split_acceptance:** Patient-disjoint allocation and recorded independent check status.
   High-burden patients in validation/test are one each. Package acceptance establishes
   reproducible data contracts, not predictive or clinical validity. Train-only model
   preprocessing is not_applicable because model training has not started.

Recommended four-minute use: cohort flow (60 s); time rules and a simplified score panel
(80 s); laboratory coverage plus split/acceptance takeaways and limitations (80 s).
Keep detailed cleaning and component tables as backup. Leave 20 s spare.

## Figure and table files

- **01_cohort_flow:** [PNG](../../figures/midterm_2026-10-05/01_cohort_flow.png), [SVG](../../figures/midterm_2026-10-05/01_cohort_flow.svg), [01_cohort_flow.csv](../../tables/midterm_2026-10-05/01_cohort_flow.csv), [01_exclusion_reasons.csv](../../tables/midterm_2026-10-05/01_exclusion_reasons.csv), [01_cohort_stage_counts.csv](../../tables/midterm_2026-10-05/01_cohort_stage_counts.csv)
- **02_time_windows:** [PNG](../../figures/midterm_2026-10-05/02_time_windows.png), [SVG](../../figures/midterm_2026-10-05/02_time_windows.svg), [02_time_windows.csv](../../tables/midterm_2026-10-05/02_time_windows.csv)
- **03_labels:** [PNG](../../figures/midterm_2026-10-05/03_labels.png), [SVG](../../figures/midterm_2026-10-05/03_labels.svg), [03_label_scores.csv](../../tables/midterm_2026-10-05/03_label_scores.csv), [03_label_events.csv](../../tables/midterm_2026-10-05/03_label_events.csv)
- **04_lab_missingness:** [PNG](../../figures/midterm_2026-10-05/04_lab_missingness.png), [SVG](../../figures/midterm_2026-10-05/04_lab_missingness.svg), [04_lab_missingness.csv](../../tables/midterm_2026-10-05/04_lab_missingness.csv), [04_retained_dictionary.csv](../../tables/midterm_2026-10-05/04_retained_dictionary.csv)
- **05_cleaning:** [PNG](../../figures/midterm_2026-10-05/05_cleaning.png), [SVG](../../figures/midterm_2026-10-05/05_cleaning.svg), [05_window_rejections.csv](../../tables/midterm_2026-10-05/05_window_rejections.csv), [05_window_rejections_by_item.csv](../../tables/midterm_2026-10-05/05_window_rejections_by_item.csv), [05_cleaning_by_feature.csv](../../tables/midterm_2026-10-05/05_cleaning_by_feature.csv)
- **06_split_acceptance:** [PNG](../../figures/midterm_2026-10-05/06_split_acceptance.png), [SVG](../../figures/midterm_2026-10-05/06_split_acceptance.svg), [06_split.csv](../../tables/midterm_2026-10-05/06_split.csv), [06_acceptance.csv](../../tables/midterm_2026-10-05/06_acceptance.csv)
