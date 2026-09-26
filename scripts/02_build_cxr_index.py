"""Entry point for section 3 of docs/Data_Processing_and_Cleaning.md."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cohort.build_cxr_index import build_cxr_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the local CXR DICOM index")
    parser.add_argument("--paths", type=Path, default=ROOT / "configs/paths.local.json")
    parser.add_argument("--imaging", type=Path, default=ROOT / "configs/imaging.json")
    args = parser.parse_args()
    return build_cxr_index(ROOT, args.paths.resolve(), args.imaging.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
