"""Publish formal burden labels from the frozen, evidence-linked audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cohort.build_labels import build_burden_labels  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build formal burden labels using the frozen endpoint decision"
    )
    for name in ("paths", "cohort", "ehr", "imaging"):
        default = "paths.local" if name == "paths" else name
        parser.add_argument(
            f"--{name}", type=Path, default=ROOT / f"configs/{default}.json"
        )
    args = parser.parse_args()
    return build_burden_labels(
        ROOT,
        *(
            getattr(args, name).resolve()
            for name in ("paths", "cohort", "ehr", "imaging")
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
