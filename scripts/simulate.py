#!/usr/bin/env python3
"""Ask the CR API to simulate pricing decisions for local basic-info cards.

    python3 scripts/simulate.py

Reads ``config/simulate.json``. Cards and outcomes must already exist from
``scripts/build_prepro.py``. Responses are written under ``outputs/simulate/``
and are not committed. Requires ``CR_API_KEY`` in the environment or ``.env``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from cr_api import CRError  # noqa: E402
from popbench.simulate.run import run_simulation  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Simulate pricing answers through the CR API.")
    parser.add_argument("config", nargs="?", default=str(ROOT / "config" / "simulate.json"))
    parser.add_argument("--limit-subjects", type=int, default=None)
    parser.add_argument("--model", default=None, help="CR text model id; default is the config value")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.limit_subjects is not None:
        config["limit_subjects"] = args.limit_subjects
    if args.model:
        config["model"] = args.model
    try:
        summary = run_simulation(ROOT, config)
    except (OSError, ValueError, KeyError, CRError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
