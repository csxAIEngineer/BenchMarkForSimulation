#!/usr/bin/env python3
"""Fetch Twin-2K-500, Twin-2K-500-Mega-Study, and SimBench into ``data/``.

Examples:
    python scripts/fetch_datasets.py twin2k_500 --limit 2
    python scripts/fetch_datasets.py simbench --split SimBenchPop --limit 20
    python scripts/fetch_datasets.py all
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from popbench.cli import main  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv or argv[0] not in {"fetch", "-h", "--help"}:
        argv = ["fetch", *argv]
    raise SystemExit(main(argv))
