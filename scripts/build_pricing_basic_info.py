#!/usr/bin/env python3
"""Build or verify exact basic-info persona cards.

From the repository root, with the committed source lock already present:

    pip install -e ".[dev]"
    python scripts/build_pricing_basic_info.py

That command downloads missing sources, checks every cached file against
source-lock.json, and writes cards. It does not create or replace the lock.

    python scripts/build_pricing_basic_info.py --verify

recomputes hashes of the committed artifacts and does not refresh them.

    python scripts/build_pricing_basic_info.py --init-source-lock

is the only command that writes source-lock.json. It is for pinning a new
upstream snapshot, not for reproducing the committed sample.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from popbench.pricing_basic_info.pipeline import SourceIntegrityError, run_build, verify_outputs  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or verify pricing basic-info cards.")
    parser.add_argument("config", nargs="?", default=str(ROOT / "config" / "pricing_basic_info.json"))
    parser.add_argument(
        "--verify",
        action="store_true",
        help="recompute hashes of committed artifacts; do not download or rewrite",
    )
    parser.add_argument(
        "--init-source-lock",
        action="store_true",
        help="write source-lock.json from the current source bytes; not the reproduction path",
    )
    args = parser.parse_args(argv)
    if args.verify and args.init_source_lock:
        print("error: --verify does not write a lock", file=sys.stderr)
        return 2
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = ROOT / str(config["output_dir"])
    try:
        if args.verify:
            verify_outputs(output)
            print(json.dumps({"verified": True, "output_dir": str(config["output_dir"])}, indent=2))
            return 0
        manifest = run_build(ROOT, config, init_source_lock=args.init_source_lock)
    except (SourceIntegrityError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"seed": manifest["seed"], "manifest_sha256": manifest["manifest_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
