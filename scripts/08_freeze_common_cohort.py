"""Freeze a versioned common cohort from published labels and reviewed image QC."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cohort.freeze_cohort import freeze_common_cohort  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify frozen labels and image QC, then freeze an immutable common cohort"
    )
    for name in ("paths", "cohort", "ehr", "imaging"):
        default = "paths.local" if name == "paths" else name
        parser.add_argument(
            f"--{name}", type=Path, default=ROOT / f"configs/{default}.json"
        )
    parser.add_argument(
        "--processing", type=Path, default=ROOT / "configs/image_processing_v1.json"
    )
    parser.add_argument("--release-version", default=None)
    args = parser.parse_args()
    return freeze_common_cohort(
        ROOT,
        args.paths.resolve(),
        args.cohort.resolve(),
        args.ehr.resolve(),
        args.imaging.resolve(),
        args.processing.resolve(),
        args.release_version,
    )


if __name__ == "__main__":
    raise SystemExit(main())
