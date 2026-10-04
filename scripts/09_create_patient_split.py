"""Publish a frozen patient-level split of the verified common cohort."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cohort.create_patient_split import create_patient_split  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze patient-level train/validation/test assignments with high-burden coverage"
    )
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
    args = parser.parse_args()
    return create_patient_split(
        ROOT,
        *(
            getattr(args, name).resolve()
            for name in ("paths", "cohort", "ehr", "imaging")
        ),
        args.processing.resolve(),
        args.split_config.resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
