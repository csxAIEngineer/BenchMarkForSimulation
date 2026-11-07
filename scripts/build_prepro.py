#!/usr/bin/env python3
"""Build or verify exact basic-info persona cards.

From the repository root, with the committed source lock already present:

    pip install -e ".[dev]"
    python3 scripts/build_prepro.py

That command downloads missing sources, checks every cached file against
source-lock.json, and writes cards. It does not create or replace the lock.
Use ``python3``, not ``python``: this repository's scripts and tests are
run with Python 3, and a bare ``python`` command is not assumed to exist.

    python3 scripts/build_prepro.py --verify

recomputes hashes of the local outputs and does not refresh them.
Cards, samples, and outcomes are written under the configured output and
cache directories. They are not part of the git tree.

    python3 scripts/build_prepro.py --init-source-lock

is the only command that writes source-lock.json. It is for pinning a new
upstream snapshot, not for reproducing an existing lock.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from popbench.prepro.pipeline import (  # noqa: E402
    SourceIntegrityError,
    lock_path_from_config,
    run_build,
    verify_outputs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or verify pricing basic-info cards.")
    parser.add_argument("config", nargs="?", default=str(ROOT / "config" / "prepro.json"))
    parser.add_argument(
        "--verify",
        action="store_true",
        help="recompute hashes of local outputs; do not download or rewrite",
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
    lock = lock_path_from_config(ROOT, config)
    try:
        if args.verify:
            report = verify_outputs(output, cache=ROOT / str(config["cache_dir"]), lock=lock)
            print(json.dumps({"verified": True, **report}, indent=2))
            return 0
        manifest = run_build(ROOT, config, init_source_lock=args.init_source_lock)
    except (SourceIntegrityError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"seed": manifest["seed"], "manifest_sha256": manifest["manifest_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
