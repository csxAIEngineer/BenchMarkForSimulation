#!/usr/bin/env python3
"""Build exact basic-info persona cards and a reproducible pricing sample.

From the repository root:

    python scripts/build_pricing_basic_info.py

The command reads config/pricing_basic_info.json. It downloads the pinned
public files into data/pricing_basic_info/ (gitignored) and writes cards,
the sample, distribution counts, and hashes under experiments/pricing-basic-info/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from popbench.pricing_basic_info.pipeline import run_build  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    config_path = ROOT / "config" / "pricing_basic_info.json"
    if argv:
        config_path = Path(argv[0])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = run_build(ROOT, config)
    print(json.dumps({"seed": manifest["seed"], "outputs": manifest["output_dir"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
