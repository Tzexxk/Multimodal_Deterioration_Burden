"""Audit V2 candidates and six components for both prediction windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cohort.audit_burden_labels import audit_burden_labels  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit V2 deterioration burden at 48 and 72 hours; do not freeze H"
    )
    for name in ("paths", "cohort", "ehr", "imaging"):
        default = "paths.local" if name == "paths" else name
        parser.add_argument(
            f"--{name}", type=Path, default=ROOT / f"configs/{default}.json"
        )
    args = parser.parse_args()
    return audit_burden_labels(
        ROOT,
        *(
            getattr(args, name).resolve()
            for name in ("paths", "cohort", "ehr", "imaging")
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
