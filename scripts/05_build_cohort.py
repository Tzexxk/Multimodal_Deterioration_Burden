"""Entry point for the section 9 candidate cohort audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cohort.build_cohort import build_candidate_cohort  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build candidate 6-48h cohort audit")
    parser.add_argument("--paths", type=Path, default=ROOT / "configs/paths.local.json")
    parser.add_argument("--cohort", type=Path, default=ROOT / "configs/cohort.json")
    parser.add_argument("--mode", choices=["candidate"], default="candidate")
    args = parser.parse_args()
    return build_candidate_cohort(ROOT, args.paths.resolve(), args.cohort.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
