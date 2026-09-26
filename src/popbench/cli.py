"""Command line: fetch the cache, run a config, or score a saved run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from popbench import __version__
from popbench.config import ConfigError, load_config
from popbench.fullsurvey import run_full_survey
from popbench.interview import InterviewError, build_interview, run_nemotron_interview
from popbench.personas import NEMOTRON_SHARD_COUNT, PersonaError, cache_nemotron_shards
from popbench.twin2k import DatasetError, load_twin2k


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="popbench",
        description="Simulate population answers and score them against Twin-2K-500.",
    )
    parser.add_argument("--version", action="version", version=f"popbench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download Twin-2K-500 and a Nemotron-USA shard")
    fetch.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="cache root (default: data/)",
    )
    fetch.add_argument(
        "--nemotron-shards",
        type=int,
        default=1,
        help=f"how many Nemotron-USA parquet shards to cache (1-{NEMOTRON_SHARD_COUNT})",
    )
    fetch.set_defaults(func=cmd_fetch)

    run = sub.add_parser("run", help="load a config and the cached human data")
    run.add_argument("--config", type=Path, required=True)
    run.set_defaults(func=cmd_run)

    score = sub.add_parser("score", help="score a saved run directory")
    score.add_argument("--run", type=Path, required=True, dest="run_dir")
    score.set_defaults(func=cmd_score)

    build = sub.add_parser("build-interview", help="export the 50-turn Twin-2K interview")
    build.add_argument("--data-dir", type=Path, default=Path("data"))
    build.add_argument("--n", type=int, default=50)
    build.add_argument("--seed", type=int, default=0)
    build.set_defaults(func=cmd_build_interview)

    answer = sub.add_parser("answer-interview", help="answer the Nemotron panel and score it")
    answer.add_argument("--data-dir", type=Path, default=Path("data"))
    answer.add_argument("--run", type=Path, default=Path("runs/interview-v0"), dest="run_dir")
    answer.add_argument("--workers", type=int, default=4)
    answer.add_argument(
        "--full",
        action="store_true",
        help="ask every closed-ended Twin-2K item, not the 50-item interview",
    )
    answer.set_defaults(func=cmd_answer_interview)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, DatasetError, PersonaError, InterviewError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def cmd_fetch(args: argparse.Namespace) -> int:
    data_dir = args.data_dir
    if not data_dir.is_absolute():
        data_dir = Path.cwd() / data_dir
    data = load_twin2k(data_dir)
    print(
        "Twin-2K-500 "
        f"revision {data.revision[:12]} "
        f"respondents {data.n_respondents} "
        f"catalog {len(data.catalog)} "
        f"wave1_3 columns {data.wave1_3.shape[1]} "
        f"wave4 columns {data.wave4.shape[1]}"
    )
    print(f"  cache {data.root}")
    shards = cache_nemotron_shards(data_dir, n_shards=args.nemotron_shards)
    print(f"Nemotron-Personas-USA shards cached {len(shards)} of {NEMOTRON_SHARD_COUNT}")
    for path in shards:
        print(f"  {path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    data = load_twin2k(config.data_dir)
    print(
        "Loaded Twin-2K-500 "
        f"({data.n_respondents} respondents, {len(data.catalog)} catalog entries) "
        f"from {data.root}"
    )
    print(
        "Config: "
        f"suite={config.suite} "
        f"persona={config.persona_condition} "
        f"n_people={config.effective_n_people} "
        f"item_cap={config.item_cap} "
        f"seed={config.seed}"
    )
    print("Model simulation is not implemented yet. No API calls were made.")
    return 0


def cmd_build_interview(args: argparse.Namespace) -> int:
    data_dir = args.data_dir if args.data_dir.is_absolute() else Path.cwd() / args.data_dir
    out = build_interview(data_dir, n=args.n, seed=args.seed)
    print(f"Wrote the 50-turn interview to {out}")
    return 0


def cmd_answer_interview(args: argparse.Namespace) -> int:
    data_dir = args.data_dir if args.data_dir.is_absolute() else Path.cwd() / args.data_dir
    run_dir = args.run_dir if args.run_dir.is_absolute() else Path.cwd() / args.run_dir
    if args.full:
        if args.run_dir == Path("runs/interview-v0"):
            run_dir = Path.cwd() / "runs" / "interview-full"
        out = run_full_survey(data_dir, run_dir, workers=args.workers)
    else:
        out = run_nemotron_interview(data_dir, run_dir, workers=args.workers)
    print(f"Wrote the comparison to {out / 'report.md'}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    run_dir = args.run_dir
    if not run_dir.is_dir():
        print(f"error: no run directory at {run_dir}", file=sys.stderr)
        return 1
    print(
        "Scoring is not implemented yet. "
        f"Saved files in {run_dir} were left untouched."
    )
    return 0
