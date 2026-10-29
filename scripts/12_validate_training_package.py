"""Independently validate the frozen training package and publish acceptance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.validate_training_package import run_validation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, filename in {
        "paths": "paths.local",
        "cohort": "cohort",
        "ehr": "ehr",
        "imaging": "imaging",
        "processing": "image_processing_v1",
        "split-config": "patient_split_v1",
        "lab-config": "lab_processing_v1",
        "feature-config": "ehr_features_v1",
        "export-config": "training_export_v1",
        "validation-config": "training_validation_v1",
    }.items():
        parser.add_argument(
            f"--{name}", type=Path, default=ROOT / f"configs/{filename}.json"
        )
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    return run_validation(ROOT, args)


if __name__ == "__main__":
    raise SystemExit(main())
