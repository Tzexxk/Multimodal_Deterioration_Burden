"""Entry point for section 5 of docs/Data_Processing_and_Cleaning.md."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cohort.match_admissions import match_admissions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Match selected CXR studies to admissions"
    )
    parser.add_argument("--paths", type=Path, default=ROOT / "configs/paths.local.json")
    parser.add_argument("--cohort", type=Path, default=ROOT / "configs/cohort.json")
    args = parser.parse_args()
    return match_admissions(ROOT, args.paths.resolve(), args.cohort.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
