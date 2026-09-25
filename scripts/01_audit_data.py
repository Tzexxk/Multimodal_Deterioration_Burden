"""Entry point for section 2 of docs/Data_Processing_and_Cleaning.md."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cohort.audit_raw_data import run_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only raw MIMIC data audit")
    parser.add_argument("--paths", type=Path, default=ROOT / "configs/paths.local.json")
    parser.add_argument("--imaging", type=Path, default=ROOT / "configs/imaging.json")
    parser.add_argument("--ehr", type=Path, default=ROOT / "configs/ehr.json")
    args = parser.parse_args()
    return run_audit(ROOT, args.paths.resolve(), args.imaging.resolve(), args.ehr.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
