"""Audit V2 candidates and six components for both prediction windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cohort.audit_burden_labels import (  # noqa: E402
    audit_burden_labels,
    freeze_endpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit V2 evidence or freeze an explicitly reviewed endpoint"
    )
    for name in ("paths", "cohort", "ehr", "imaging"):
        default = "paths.local" if name == "paths" else name
        parser.add_argument(
            f"--{name}", type=Path, default=ROOT / f"configs/{default}.json"
        )
    parser.add_argument("--mode", choices=("audit", "freeze"), default="audit")
    parser.add_argument("--decision-review", type=Path)
    args = parser.parse_args()
    if (args.mode == "freeze") != bool(args.decision_review):
        parser.error("--decision-review is required only in freeze mode")
    if args.mode == "freeze":
        return freeze_endpoint(
            ROOT,
            *(
                getattr(args, name).resolve()
                for name in ("paths", "cohort", "ehr", "imaging")
            ),
            args.decision_review.resolve(),
        )
    return audit_burden_labels(
        ROOT,
        *(
            getattr(args, name).resolve()
            for name in ("paths", "cohort", "ehr", "imaging")
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
