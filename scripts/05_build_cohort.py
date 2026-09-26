"""Entry point for a window-specific candidate cohort audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cohort.build_cohort import build_candidate_cohort  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build candidate cohort audit")
    parser.add_argument("--paths", type=Path, default=ROOT / "configs/paths.local.json")
    parser.add_argument("--cohort", type=Path, default=ROOT / "configs/cohort.json")
    parser.add_argument("--mode", choices=["candidate"], default="candidate")
    parser.add_argument(
        "--max-hours",
        type=int,
        required=True,
        help="Prediction window upper bound; must equal cohort.json prediction.max_hours",
    )
    args = parser.parse_args()
    return build_candidate_cohort(
        ROOT, args.paths.resolve(), args.cohort.resolve(), args.max_hours
    )


if __name__ == "__main__":
    raise SystemExit(main())
