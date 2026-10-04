"""Export the frozen training directory and a matching portable ZIP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.export_training_package import run_export  # noqa: E402


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
    }.items():
        parser.add_argument(
            f"--{name}", type=Path, default=ROOT / f"configs/{filename}.json"
        )
    return run_export(ROOT, parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
