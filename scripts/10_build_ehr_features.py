"""Audit, review, and explicitly freeze laboratory rules before EHR aggregation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ehr.build_ehr_features import run_step10  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "review", "freeze"), required=True)
    for name in ("paths", "cohort", "ehr", "imaging"):
        default = "paths.local" if name == "paths" else name
        parser.add_argument(
            f"--{name}", type=Path, default=ROOT / f"configs/{default}.json"
        )
    parser.add_argument(
        "--processing", type=Path, default=ROOT / "configs/image_processing_v1.json"
    )
    parser.add_argument(
        "--split-config", type=Path, default=ROOT / "configs/patient_split_v1.json"
    )
    parser.add_argument(
        "--lab-config", type=Path, default=ROOT / "configs/lab_processing_v1.json"
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        nargs="+",
        help="Completed, cumulative versioned review CSVs",
    )
    parser.add_argument(
        "--dictionary-version",
        help="Required immutable dictionary identity in freeze mode",
    )
    args = parser.parse_args()
    if args.mode != "audit" and not args.reviews:
        parser.error("review and freeze modes require --reviews")
    if args.mode == "freeze" and not args.dictionary_version:
        parser.error("freeze mode requires --dictionary-version")
    if args.mode != "freeze" and args.dictionary_version:
        parser.error("--dictionary-version is only valid in freeze mode")
    if args.mode == "audit" and args.reviews:
        parser.error("audit mode does not accept review decisions")
    return run_step10(
        ROOT,
        args.mode,
        [getattr(args, k).resolve() for k in ("paths", "cohort", "ehr", "imaging")],
        args.processing.resolve(),
        args.split_config.resolve(),
        args.lab_config.resolve(),
        [p.resolve() for p in args.reviews or []],
        args.dictionary_version,
    )


if __name__ == "__main__":
    raise SystemExit(main())
