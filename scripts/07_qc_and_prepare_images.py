"""Prepare the selected DICOMs and publish evidence-linked image QC."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.imaging.prepare_images import prepare_images  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare formal-label images and publish deterministic QC and review previews"
    )
    for name in ("paths", "cohort", "ehr", "imaging"):
        default = "paths.local" if name == "paths" else name
        parser.add_argument(
            f"--{name}", type=Path, default=ROOT / f"configs/{default}.json"
        )
    parser.add_argument(
        "--processing", type=Path, default=ROOT / "configs/image_processing_v1.json"
    )
    parser.add_argument("--reviews", type=Path, default=None)
    args = parser.parse_args()
    return prepare_images(
        ROOT,
        *(
            getattr(args, name).resolve()
            for name in ("paths", "cohort", "ehr", "imaging")
        ),
        args.processing.resolve(),
        args.reviews.resolve() if args.reviews else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
