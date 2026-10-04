"""Audit and publish pre-imaging EHR features without changing Step 10A."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ehr.step10b import run_step10b  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "build"), required=True)
    parser.add_argument("--dictionary-version", required=True)
    parser.add_argument("--feature-set-version")
    for name, filename in {
        "paths": "paths.local",
        "cohort": "cohort",
        "ehr": "ehr",
        "imaging": "imaging",
        "processing": "image_processing_v1",
        "split-config": "patient_split_v1",
        "lab-config": "lab_processing_v1",
        "feature-config": "ehr_features_v1",
    }.items():
        parser.add_argument(
            f"--{name}", type=Path, default=ROOT / f"configs/{filename}.json"
        )
    args = parser.parse_args()
    if (args.mode == "build") != bool(args.feature_set_version):
        parser.error("--feature-set-version is required only in build mode")
    return run_step10b(ROOT, args)


if __name__ == "__main__":
    raise SystemExit(main())
