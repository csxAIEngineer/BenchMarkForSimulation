#!/usr/bin/env python3
"""Score local CR simulation responses against held-out human outcomes.

    python3 scripts/evaluate.py

Does not call a model. Writes the report to the path in ``config/evaluate.json``.
That path is under ``outputs/`` and is not committed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from popbench.evaluate.score import score_run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate local simulation responses.")
    parser.add_argument("config", nargs="?", default=str(ROOT / "config" / "evaluate.json"))
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    try:
        report = score_run(ROOT, config)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
