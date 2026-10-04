"""Create aggregate presentation figures from an accepted local training release.

This is a reporting utility, not a data-processing pipeline step. Raw inputs,
patient-level outputs, frozen releases and model preprocessing are never written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

RELEASE = "burden_v2_h72_r01"
PACKAGE = f"data/processed/training/{RELEASE}"
AUDIT = "outputs/audit/burden_v2"
EHR_AUDIT = f"{AUDIT}/ehr/ehr_audit_v1"
COMPONENTS = {
    "new_icu_ccu_transfer": "New ICU/CCU transfer",
    "new_stepdown_transfer": "New stepdown transfer",
    "new_invasive_ventilation": "New invasive ventilation",
    "new_severe_hypoxemia": "New severe hypoxemia",
    "new_iv_vaso_inotrope_infusion": "New IV vasoactive infusion",
    "in_hospital_death": "In-hospital death",
}
BLUE, TEAL, ORANGE = "#24558A", "#187F7A", "#C67528"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def label_tables(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count studies and independent positive patients without summing overlaps."""
    if samples.empty or samples.sample_id.duplicated().any():
        raise ValueError("Samples must be nonempty with unique sample keys")
    if not samples.burden_score.isin(range(9)).all():
        raise ValueError("Invalid burden score")
    counts = samples.burden_score.value_counts().reindex(range(9), fill_value=0)
    scores = pd.DataFrame(
        {"score": range(9), "studies": counts.to_numpy(), "denominator": len(samples)}
    )
    scores["percent"] = 100 * scores.studies / len(samples)
    events = []
    for field, label in {
        **COMPONENTS,
        "phys_death": "Physiologic deterioration / death",
        "high_burden": "High burden (score >= 3)",
    }.items():
        if not bool(samples[field].isin([0, 1]).all()):
            raise ValueError(f"Invalid component: {field}")
        positive = samples.loc[samples[field] == 1]
        events.append(
            dict(
                component=field,
                label=label,
                studies=len(positive),
                patients=positive.subject_id.nunique(),
                study_denominator=len(samples),
                patient_denominator=samples.subject_id.nunique(),
            )
        )
    return scores, pd.DataFrame(events)


def split_table(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.groupby("subject_id").split.nunique().gt(1).any():
        raise ValueError("A patient crosses split boundaries")
    if not samples.split.isin(["train", "validation", "test"]).all():
        raise ValueError("Unknown split")
    rows = []
    for split in ["train", "validation", "test"]:
        group = samples.loc[samples.split == split]
        rows.append(
            dict(
                split=split,
                studies=len(group),
                admissions=group.hadm_id.nunique(),
                patients=group.subject_id.nunique(),
                phys_death_patients=group.loc[
                    group.phys_death == 1
                ].subject_id.nunique(),
                high_burden_patients=group.loc[
                    group.high_burden == 1
                ].subject_id.nunique(),
            )
        )
    return pd.DataFrame(rows)


def missingness_table(samples: pd.DataFrame, ehr: pd.DataFrame) -> pd.DataFrame:
    if ehr.sample_id.duplicated().any() or set(samples.sample_id) != set(ehr.sample_id):
        raise ValueError("EHR and sample keys disagree")
    joined = samples[["sample_id", "subject_id", "split"]].merge(
        ehr, on="sample_id", validate="one_to_one"
    )
    rows = []
    for column in sorted(
        c for c in ehr.columns if isinstance(c, str) and c.endswith("_last_24h")
    ):
        for split in ["all", "train", "validation", "test"]:
            group = joined if split == "all" else joined.loc[joined.split == split]
            present = group.loc[group[column].notna()]
            rows.append(
                dict(
                    feature=column.removesuffix("_last_24h"),
                    split=split,
                    studies=len(group),
                    observed_studies=len(present),
                    observed_patients=present.subject_id.nunique(),
                    missing_rate=1 - len(present) / len(group)
                    if len(group)
                    else np.nan,
                )
            )
    return pd.DataFrame(rows)


def rejection_table(window: pd.DataFrame) -> pd.DataFrame:
    """Keep the joint categories; distinct source counts are not additive."""
    joint = window.groupby(["availability_reason", "ownership_reason"], as_index=False)[
        ["study_associations"]
    ].sum()
    if not isinstance(joint, pd.DataFrame):
        raise TypeError("Joint association counts must be a DataFrame")
    if (joint.study_associations < 0).any():
        raise ValueError("Negative association count")
    return joint


def save_figure(fig, path: Path, footer: str) -> None:
    fig.text(0.03, 0.025, footer, fontsize=10, color="#46556A", va="bottom")
    fig.savefig(path.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(path.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def style_axis(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E5EAF0", linewidth=0.8)


def bar_labels(ax, bars, *, percent: bool = False) -> None:
    ax.bar_label(bars, fmt="%.1f%%" if percent else "%d", padding=5, fontsize=11)


def render_figures(tables: dict[str, pd.DataFrame], folder: Path) -> None:
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 13, "svg.fonttype": "none"}
    )
    flow = tables["01_cohort_flow"]
    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.set_title("A common cohort for three model comparisons", fontsize=21, pad=18)
    for i, (_, row) in enumerate(flow.iterrows()):
        y = 0.86 - i * 0.15
        ax.add_patch(
            FancyBboxPatch(
                (0.08, y - 0.075),
                0.58,
                0.11,
                boxstyle="round,pad=0.012",
                facecolor="#E9F1F8",
                edgecolor=BLUE,
            )
        )
        ax.text(
            0.37,
            y - 0.02,
            f"{row['label']}\n{row['count']:,} {row['unit']}",
            ha="center",
            va="center",
            fontsize=14,
            color=BLUE,
        )
        if i < len(flow) - 1:
            ax.annotate(
                "",
                xy=(0.37, y - 0.115),
                xytext=(0.37, y - 0.085),
                arrowprops=dict(arrowstyle="->", color=BLUE),
            )
        note = str(row["note"])
        if note:
            ax.text(0.71, y - 0.02, note, va="center", fontsize=12)
    save_figure(
        fig,
        folder / "01_cohort_flow",
        "Counts change from images to studies. Final cohort: 473 studies / 303 admissions / 222 patients.",
    )

    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    ax.set(xlim=(-29, 78), ylim=(-1, 4.5))
    ax.axis("off")
    ax.set_title(
        "Use only information available at the image time", fontsize=21, pad=18
    )
    ax.plot([-24, 0], [2, 2], lw=14, color=TEAL, solid_capstyle="butt")
    ax.plot([0, 72], [2, 2], lw=14, color=BLUE, solid_capstyle="butt")
    for x, text in [(-24, "t0 - 24 h"), (0, "t0: CXR"), (72, "t0 + 72 h")]:
        ax.plot([x, x], [1.7, 2.3], color="#26384C", lw=2)
        ax.text(x, 1.42, text, ha="center", fontsize=14)
    ax.text(-12, 2.6, "EHR lookback\n[-24 h, 0 h]", ha="center", color=TEAL)
    ax.text(
        36, 2.6, "Outcome observation\n(0 h, observation end]", ha="center", color=BLUE
    )
    ax.text(
        -25,
        0.45,
        "Lab charttime inside lookback\ncharttime <= storetime <= t0\nPatient + admission ownership required",
        fontsize=13,
    )
    ax.text(
        18,
        0.45,
        "Observation ends by 72 h or discharge\nEarly events remain included\nFuture outcomes are labels only",
        fontsize=13,
    )
    ax.text(
        22,
        3.65,
        "Schematic method diagram; no individual patient timeline",
        ha="center",
        fontsize=12,
    )
    save_figure(
        fig,
        folder / "02_time_windows",
        "Patient-level splitting prevents cross-group reuse. Imputation, scaling and category fitting remain train-only future work.",
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 7.5), gridspec_kw={"width_ratios": [1, 1.65]}
    )
    scores = tables["03_label_scores"]
    bars = axes[0].bar(scores.score, scores.studies, color=BLUE)
    axes[0].bar_label(bars, padding=4, fontsize=11)
    axes[0].set(
        title="Burden score",
        xlabel="Score (defined range: 0-8)",
        ylabel="Studies",
        xticks=range(9),
        ylim=(0, max(scores.studies) * 1.17),
    )
    events = tables["03_label_events"].iloc[:6]
    y = np.arange(len(events))
    axes[1].barh(y - 0.18, events.studies, height=0.34, color=BLUE, label="Studies")
    axes[1].barh(y + 0.18, events.patients, height=0.34, color=TEAL, label="Patients")
    for i, (_, row) in enumerate(events.iterrows()):
        axes[1].text(
            row.studies + 0.4, i - 0.18, str(row.studies), va="center", fontsize=11
        )
        axes[1].text(
            row.patients + 0.4, i + 0.18, str(row.patients), va="center", fontsize=11
        )
    axes[1].set(
        yticks=y,
        yticklabels=events.label,
        xlabel="Positive counts",
        title="Recorded outcome components",
    )
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, max(events.studies) * 1.2)
    axes[1].legend(loc="lower right", frameon=False)
    style_axis(axes[1])
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.18, top=0.87, wspace=0.85)
    fig.suptitle("Final cohort labels: strong class imbalance", fontsize=21)
    save_figure(
        fig,
        folder / "03_labels",
        "Denominators: 473 studies / 222 patients. Components overlap; counts must not be added. High burden: 12 studies / 8 patients.",
    )

    missing = tables["04_lab_missingness"]
    missing = missing.loc[missing.split == "all"].sort_values(
        ["missing_rate", "feature"]
    )
    fig, ax = plt.subplots(figsize=(13, 10))
    colors = [
        TEAL if v < 0.3 else BLUE if v <= 0.5 else ORANGE for v in missing.missing_rate
    ]
    bars = ax.barh(
        missing.feature.str.replace("_", " "), missing.missing_rate * 100, color=colors
    )
    for bar, (_, row) in zip(bars, missing.iterrows(), strict=True):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{row.missing_rate:.1%}   (observed n={row.observed_studies})",
            va="center",
            fontsize=11,
        )
    ax.set(
        xlim=(0, 110),
        xlabel="Studies without an accepted value in the pre-CXR window (%)",
        title="Coverage of all 25 retained laboratory features",
    )
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.invert_yaxis()
    style_axis(ax)
    fig.subplots_adjust(left=0.21, right=0.98, top=0.93, bottom=0.13)
    save_figure(
        fig,
        folder / "04_lab_missingness",
        "Denominator: 473 studies; missingness is based on last_24h. Descriptive coverage only; retained rules are unchanged.\n"
        "Colors: <30%, 30-50%, >50%. No automatic feature selection or imputation is applied.",
    )

    joint = tables["05_window_rejections"]
    clean = tables["05_cleaning_by_feature"]
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 8.5), gridspec_kw={"width_ratios": [1.6, 1]}
    )
    labels = joint.apply(
        lambda r: (
            f"{r.availability_reason.replace('_', ' ')}\n{r.ownership_reason.replace('_', ' ')}"
        ),
        axis=1,
    )
    bars = axes[0].barh(labels, joint.study_associations, color=ORANGE)
    bar_labels(axes[0], bars)
    axes[0].invert_yaxis()
    axes[0].set(
        title="Window rejections: joint reasons", xlabel="Study-window associations"
    )
    disposition = clean.groupby("disposition").study_associations.sum()
    bars = axes[1].bar(
        disposition.index.str.replace("_", " "),
        disposition.values,
        color=[ORANGE, TEAL][: len(disposition)],
    )
    axes[1].bar_label(bars, padding=4, fontsize=12)
    axes[1].set(
        title="Approved-rule processing",
        ylabel="Feature-window associations",
        ylim=(0, max(disposition) * 1.15),
    )
    axes[1].tick_params(axis="x", labelsize=11)
    style_axis(axes[0])
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.18, top=0.85, wspace=0.35)
    fig.suptitle(
        "Audit rejections and reviewed cleaning are separate stages", fontsize=20
    )
    save_figure(
        fig,
        folder / "05_cleaning",
        "Left: all observed items failing availability or ownership, before approved feature rules. Right: reviewed feature processing.\n"
        "These panels have different scopes; do not subtract or combine them. Source counts are preserved per input group in CSV.",
    )

    split = tables["06_split"]
    checks = tables["06_acceptance"]
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 7.5), gridspec_kw={"width_ratios": [1, 1.4]}
    )
    x = np.arange(3)
    axes[0].bar(x - 0.19, split.studies, width=0.36, color=BLUE, label="Studies")
    axes[0].bar(x + 0.19, split.patients, width=0.36, color=TEAL, label="Patients")
    for i, (_, row) in enumerate(split.iterrows()):
        axes[0].text(
            i - 0.19, row.studies + 6, str(row.studies), ha="center", fontsize=12
        )
        axes[0].text(
            i + 0.19, row.patients + 6, str(row.patients), ha="center", fontsize=12
        )
    axes[0].set(
        xticks=x,
        xticklabels=split.split.str.title(),
        ylim=(0, 390),
        ylabel="Count",
        title="Patient-disjoint groups",
    )
    axes[0].legend(frameon=False)
    axes[1].axis("off")
    table = axes[1].table(
        cellText=checks.values,
        colLabels=["Independent check", "Status"],
        cellLoc="left",
        colLoc="left",
        bbox=[0, 0.20, 1, 0.78],
        colWidths=[0.75, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("#E5EAF0")
        if r == 0:
            cell.set_facecolor("#E9F1F8")
    axes[1].text(
        0,
        0.07,
        "High-burden patients: train {} / validation {} / test {}\n"
        "Training package accepted; model training remains pending.".format(
            *split.high_burden_patients.tolist()
        ),
        fontsize=12,
    )
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.15, top=0.85, wspace=0.22)
    fig.suptitle("Frozen split and independent package acceptance", fontsize=21)
    save_figure(
        fig,
        folder / "06_split_acceptance",
        "Patient allocation: seed 42, weights 70/15/15. Small positive counts limit evaluation stability; acceptance verifies data contracts.",
    )


def generate(root: Path, report_id: str) -> None:
    if not re.fullmatch(r"midterm_\d{4}-\d{2}-\d{2}", report_id):
        raise ValueError("Report ID must have the form midterm_YYYY-MM-DD")
    sources = [
        f"{PACKAGE}/{name}"
        for name in [
            "samples.parquet",
            "ehr_features.parquet",
            "dataset_metadata.json",
            "feature_schema.json",
        ]
    ] + [
        f"{AUDIT}/training_package_acceptance.json",
        f"{AUDIT}/cohort_flow.csv",
        f"{AUDIT}/excluded_studies.csv",
        f"{AUDIT}/split_summary.csv",
        f"{AUDIT}/lab_dictionary_freeze.json",
        f"{AUDIT}/labs/lab_audit_v1/lab_dictionary_audit.csv",
        f"{AUDIT}/endpoint_decision.json",
        f"{AUDIT}/window_comparison.csv",
        f"{AUDIT}/step_09_report.md",
        "outputs/audit/admission_match_summary.csv",
        "outputs/audit/cxr_index_summary.json",
        "data/processed/ehr/burden_v2/lab_dictionary.parquet",
        "outputs/audit/study_image_selection_summary.json",
        "outputs/audit/step_05_csv_tail_checks.csv",
        f"{EHR_AUDIT}/lab_missingness.csv",
        f"{EHR_AUDIT}/lab_window_rejections.csv",
        f"{EHR_AUDIT}/lab_cleaning_summary.csv",
        "configs/image_processing_v1.json",
        "configs/ehr_features_v1.json",
        "configs/patient_split_v1.json",
        "configs/cohort.json",
        "configs/reviews/image_qc_reviews_v5.csv",
        "configs/reviews/lab_dictionary_reviews_2026-10-05_v1_rebuild_v1.csv",
        "configs/reviews/lab_dictionary_scope_2026-10-05_v1_rebuild_v1.csv",
        "README.md",
        f"{PACKAGE}/README.md",
        "src/cohort/build_labels.py",
        "src/imaging/preprocessing.py",
        "src/ehr/step10b.py",
        "scripts/generate_midterm_report.py",
    ]
    for language in ["EN", "ZH"]:
        name = f"docs/codex/MIDTERM_DATA_PREPARATION_2026-10-05_{language}.md"
        if (root / name).exists():
            sources.append(name)
    before = {name: sha256(root / name) for name in sources}
    acceptance = load_json(root / f"{AUDIT}/training_package_acceptance.json")
    metadata = load_json(root / f"{PACKAGE}/dataset_metadata.json")
    schema = load_json(root / f"{PACKAGE}/feature_schema.json")
    if not all(
        acceptance.get(k) is True
        for k in [
            "training_package_accepted",
            "validation_passed",
            "independent_validation_completed",
        ]
    ):
        raise ValueError(
            "The local training package has not passed independent validation"
        )
    if metadata["release_id"] != RELEASE or metadata["horizon_hours"] != 72:
        raise ValueError("Unexpected frozen release")
    for name in [
        "samples.parquet",
        "ehr_features.parquet",
        "feature_schema.json",
        "dataset_metadata.json",
    ]:
        if (
            before[f"{PACKAGE}/{name}"]
            != acceptance["binding"]["package_file_sha256"][name]
        ):
            raise ValueError(f"Accepted package file changed: {name}")
    evidence = {
        str(Path(k)): v
        for k, v in acceptance["binding"]["evidence_file_sha256"].items()
    }
    for name in sources:
        key = str(root / name)
        if key in evidence and before[name] != evidence[key]:
            raise ValueError(f"Acceptance-bound evidence changed: {name}")
    samples = pd.read_parquet(root / f"{PACKAGE}/samples.parquet")
    ehr = pd.read_parquet(root / f"{PACKAGE}/ehr_features.parquet")
    if (len(samples), samples.hadm_id.nunique(), samples.subject_id.nunique()) != (
        473,
        303,
        222,
    ):
        raise ValueError("Unexpected cohort counts")
    if (len(schema["numeric_features"]), len(schema["categorical_features"])) != (
        152,
        5,
    ):
        raise ValueError("Unexpected feature schema")
    scores, events = label_tables(samples)
    splits = split_table(samples)
    if splits.patients.tolist() != [156, 33, 33] or splits.studies.tolist() != [
        337,
        69,
        67,
    ]:
        raise ValueError("Unexpected split counts")
    missing = missingness_table(samples, ehr)
    dictionary = pd.read_parquet(
        root / "data/processed/ehr/burden_v2/lab_dictionary.parquet"
    )
    retained = dictionary.loc[dictionary.keep == True].copy()  # noqa: E712
    freeze = load_json(root / f"{AUDIT}/lab_dictionary_freeze.json")
    inventory = pd.read_csv(
        root / f"{AUDIT}/labs/lab_audit_v1/lab_dictionary_audit.csv"
    )
    if (inventory.itemid.nunique(), len(dictionary)) != (368, 370):
        raise ValueError("Unexpected audited laboratory scope")
    if (len(retained), retained.itemid.nunique()) != (26, 25):
        raise ValueError("Unexpected retained laboratory branches")
    if (
        before["data/processed/ehr/burden_v2/lab_dictionary.parquet"]
        != freeze["dictionary_sha256"]
    ):
        raise ValueError("Frozen laboratory dictionary changed")
    audit_missing = pd.read_csv(root / f"{EHR_AUDIT}/lab_missingness.csv")
    check = missing.merge(audit_missing, on=["feature", "split"], validate="one_to_one")
    if len(check) != len(missing) or not np.allclose(
        check.missing_rate, check.after_missing_rate
    ):
        raise ValueError("Reconstructed missingness disagrees with the EHR audit")
    if not (check.observed_studies == check.after_studies).all():
        raise ValueError("Observed laboratory counts disagree")
    published_split = pd.read_csv(root / f"{AUDIT}/split_summary.csv")
    totals = published_split.loc[published_split.category == "total"].set_index("split")
    for row in splits.to_dict("records"):
        if any(
            row[k] != totals.loc[str(row["split"]), k]
            for k in ["studies", "patients", "admissions"]
        ):
            raise ValueError("Reconstructed split summary disagrees")
    selection = load_json(root / "outputs/audit/study_image_selection_summary.json")[
        "counts"
    ]
    index = load_json(root / "outputs/audit/cxr_index_summary.json")["counts"]
    cohort = pd.read_csv(root / f"{AUDIT}/cohort_flow.csv").set_index("stage")
    excluded = pd.read_csv(
        root / f"{AUDIT}/excluded_studies.csv", usecols=["exclusion_reason"]
    )
    if len(excluded) != cohort.loc["upstream_excluded", "studies"]:
        raise ValueError("Excluded-study count disagrees")
    flow = pd.DataFrame(
        [
            [
                "Local indexed CXR",
                index["matched_local_files"],
                "images",
                f"{selection['input_studies']:,} studies; local CXR subset",
            ],
            [
                "Frontal study selection",
                selection["selected_studies"],
                "studies",
                f"{selection['excluded_studies']} studies without frontal images",
            ],
            [
                "Unique admission match",
                cohort.loc["all_candidates", "studies"],
                "studies",
                f"{selection['selected_studies'] - cohort.loc['all_candidates', 'studies']:,} studies without admission match",
            ],
            [
                "Reviewed formal labels",
                cohort.loc["formal_labels", "studies"],
                "studies",
                f"{len(excluded):,} upstream exclusions",
            ],
            [
                "Image QC / common cohort",
                len(samples),
                "studies",
                f"{cohort.loc['qc_excluded', 'studies']} studies excluded by image QC",
            ],
        ],
        columns=["label", "count", "unit", "note"],
    )
    window = pd.read_csv(root / f"{EHR_AUDIT}/lab_window_rejections.csv")
    clean = pd.read_csv(root / f"{EHR_AUDIT}/lab_cleaning_summary.csv")
    checks = pd.DataFrame(
        [
            [name.replace("_", " "), result["status"]]
            for name, result in acceptance["checks"].items()
        ],
        columns=["check", "status"],
    )
    if any(
        v["status"] != "pass"
        for k, v in acceptance["checks"].items()
        if k != "train_only_model_preprocessing"
    ):
        raise ValueError("A mandatory acceptance check did not pass")
    tables = {
        "01_cohort_flow": flow,
        "01_exclusion_reasons": excluded.value_counts().rename("studies").reset_index(),
        "01_cohort_stage_counts": cohort.reset_index(),
        "02_time_windows": pd.DataFrame(
            [
                [
                    "lab_charttime",
                    -24,
                    0,
                    "inclusive",
                    "charttime <= storetime <= t0; ownership required",
                ],
                [
                    "outcome",
                    0,
                    72,
                    "left open, right closed",
                    "end at frozen observation_end; discharge may shorten window",
                ],
            ],
            columns=["window", "start_hours", "maximum_end_hours", "boundary", "rule"],
        ),
        "03_label_scores": scores,
        "03_label_events": events,
        "04_lab_missingness": missing,
        "05_window_rejections": rejection_table(window),
        "04_retained_dictionary": retained[
            [
                "feature_name",
                "itemid",
                "source_unit",
                "canonical_unit",
                "conversion_multiplier",
                "conversion_offset",
                "valid_min",
                "valid_max",
                "valid_range_basis",
                "specimen_rule",
                "duplicate_rule",
                "numeric_result_rule",
                "missing_unit_rule",
            ]
        ],
        "05_window_rejections_by_item": window,
        "05_cleaning_by_feature": clean,
        "06_split": splits,
        "06_acceptance": checks,
    }
    folders = {
        name: root / "reports" / name / report_id
        for name in ["figures", "tables", "manifests"]
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(folders["tables"] / f"{name}.csv", index=False, encoding="utf-8")
    render_figures(tables, folders["figures"])
    captions = """# Midterm figure captions and reproduction

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
"""
    captions += "\n## Figure and table files\n\n"
    for stem, data in [
        (
            "01_cohort_flow",
            ["01_cohort_flow", "01_exclusion_reasons", "01_cohort_stage_counts"],
        ),
        ("02_time_windows", ["02_time_windows"]),
        ("03_labels", ["03_label_scores", "03_label_events"]),
        ("04_lab_missingness", ["04_lab_missingness", "04_retained_dictionary"]),
        (
            "05_cleaning",
            [
                "05_window_rejections",
                "05_window_rejections_by_item",
                "05_cleaning_by_feature",
            ],
        ),
        ("06_split_acceptance", ["06_split", "06_acceptance"]),
    ]:
        links = [
            f"[{ext.upper()}](../../figures/{report_id}/{stem}.{ext})"
            for ext in ["png", "svg"]
        ] + [f"[{name}.csv](../../tables/{report_id}/{name}.csv)" for name in data]
        captions += f"- **{stem}:** " + ", ".join(links) + "\n"
    (folders["manifests"] / "FIGURE_GUIDE.md").write_text(captions, encoding="utf-8")
    after = {name: sha256(root / name) for name in sources}
    if before != after:
        raise ValueError(
            "A reporting input changed during generation; do not use these results"
        )
    artifacts = {
        path.relative_to(root).as_posix(): sha256(path)
        for folder in folders.values()
        for path in sorted(folder.iterdir())
        if path.is_file() and path.name != "report_manifest.json"
    }
    manifest = dict(
        report_id=report_id,
        generated_utc=datetime.now(timezone.utc).isoformat(),
        release_id=RELEASE,
        cohort_version=metadata["cohort_version"],
        split_version=metadata["split_version"],
        dictionary_version=metadata["dictionary_version"],
        feature_set_version=metadata["feature_set_version"],
        horizon_hours=72,
        acceptance_run_id=acceptance["run_id"],
        source_sha256=before,
        artifact_sha256=artifacts,
        source_stability_verified=True,
        reconstruction_checks=[
            "accepted package hashes",
            "cohort and split counts",
            "patient disjointness",
            "laboratory coverage",
        ],
        command=f"python scripts/generate_midterm_report.py --report-id {report_id}",
        confidentiality="Aggregate statistics only; no subject IDs, sample keys, individual times or images exported.",
    )
    (folders["manifests"] / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"Created six English figure pairs and {len(tables)} aggregate tables: reports/*/{report_id}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing the accepted processed release",
    )
    parser.add_argument(
        "--report-id",
        default="midterm_2026-10-05",
        help="Reporting output namespace (midterm_YYYY-MM-DD)",
    )
    args = parser.parse_args()
    generate(args.root.resolve(), args.report_id)


if __name__ == "__main__":
    main()
