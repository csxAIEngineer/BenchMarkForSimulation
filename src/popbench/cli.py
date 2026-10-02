"""Command line for the twin-2k-50 benchmark.

`dao` builds interview records, `simulate` answers them, `evaluate` scores
the saved answers. `run` does those three steps for the persona in the config.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from popbench import __version__
from popbench.config import ConfigError, load_config
from popbench.dao.interview import InterviewError, build_interview
from popbench.dao.personas import NEMOTRON_SHARD_COUNT, PersonaError, cache_nemotron_shards
from popbench.dao.twin2k import DatasetError, load_twin2k
from popbench.dao.visitors import VisitorError, build_visitors, fetch_visitors
from popbench.evaluate.interview import evaluate_interview
from popbench.evaluate.survey import evaluate_survey
from popbench.simulate.interview import run_interview
from popbench.simulate.survey import answer_survey
from popbench.simulate.visitors import run_visitors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="popbench",
        description=(
            "Build interview or ACS/ATUS visitor records, simulate answers with "
            "an LLM, and score where a human baseline exists."
        ),
    )
    parser.add_argument("--version", action="version", version=f"popbench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser(
        "fetch",
        help="download Twin-2K-500, a Nemotron-USA shard, and visitor sources",
    )
    _add_data_dir(fetch)
    fetch.add_argument(
        "--nemotron-shards",
        type=int,
        default=1,
        help=f"how many Nemotron-USA parquet shards to cache (1-{NEMOTRON_SHARD_COUNT})",
    )
    fetch.add_argument(
        "--visitors",
        action="store_true",
        help="also download ACS, ATUS, and Census visitor sources",
    )
    fetch.add_argument(
        "--visitors-only",
        action="store_true",
        help="download only the simulate-visitor sources",
    )
    fetch.set_defaults(func=cmd_fetch)

    build = sub.add_parser("build", help="build twin-2k-50 interview records")
    _add_data_dir(build)
    build.add_argument("--n", type=int, default=50)
    build.add_argument("--seed", type=int, default=0)
    build.set_defaults(func=cmd_build)

    build_visitors_cmd = sub.add_parser(
        "build-visitors",
        help="assemble ACS/ATUS/Census visitor DAO records",
    )
    _add_data_dir(build_visitors_cmd)
    build_visitors_cmd.add_argument("--n", type=int, default=50)
    build_visitors_cmd.add_argument("--seed", type=int, default=0)
    build_visitors_cmd.set_defaults(func=cmd_build_visitors)

    legacy_build = sub.add_parser("build-interview", help="alias of build")
    _add_data_dir(legacy_build)
    legacy_build.add_argument("--n", type=int, default=50)
    legacy_build.add_argument("--seed", type=int, default=0)
    legacy_build.set_defaults(func=cmd_build)

    simulate = sub.add_parser(
        "simulate",
        help="LLM answers as each person (twin-2k-50 or ACS/ATUS visitors)",
    )
    _add_data_dir(simulate)
    _add_run_dir(simulate, Path("runs/interview-v0"))
    simulate.add_argument(
        "--panel",
        choices=("nemotron", "twin2k", "visitors"),
        default="visitors",
        help="visitors = ACS/ATUS visit interview; nemotron/twin2k = twin-2k-50",
    )
    simulate.add_argument("--workers", type=int, default=4)
    simulate.add_argument(
        "--full",
        action="store_true",
        help="ask the longer closed-ended Twin-2K form instead of twin-2k-50",
    )
    simulate.add_argument("--n", type=int, default=50)
    simulate.add_argument("--seed", type=int, default=0)
    simulate.add_argument(
        "--rebuild",
        action="store_true",
        help="for --panel visitors, rebuild the ACS/ATUS panel before simulating",
    )
    simulate.set_defaults(func=cmd_simulate)

    run_visitors_cmd = sub.add_parser(
        "run-visitors",
        help="fetch ACS/ATUS, build visitors, and simulate visit interviews",
    )
    _add_data_dir(run_visitors_cmd)
    run_visitors_cmd.add_argument("--n", type=int, default=50)
    run_visitors_cmd.add_argument("--seed", type=int, default=0)
    run_visitors_cmd.add_argument("--workers", type=int, default=4)
    run_visitors_cmd.add_argument(
        "--run",
        type=Path,
        default=None,
        dest="run_dir",
        help="output directory (default: runs/visitors-n<n>-seed<seed>)",
    )
    run_visitors_cmd.set_defaults(func=cmd_run_visitors)

    evaluate = sub.add_parser("evaluate", help="score a saved run against Twin-2K")
    _add_data_dir(evaluate)
    _add_run_dir(evaluate, Path("runs/interview-v0"))
    evaluate.add_argument("--panel", choices=("nemotron", "twin2k"), default=None)
    evaluate.add_argument(
        "--full",
        action="store_true",
        help="score the longer closed-ended form",
    )
    evaluate.set_defaults(func=cmd_evaluate)

    score = sub.add_parser("score", help="alias of evaluate")
    _add_data_dir(score)
    _add_run_dir(score, Path("runs/interview-v0"))
    score.add_argument("--panel", choices=("nemotron", "twin2k"), default=None)
    score.add_argument("--full", action="store_true")
    score.set_defaults(func=cmd_evaluate)

    answer = sub.add_parser("answer-interview", help="simulate a panel, then evaluate it")
    _add_data_dir(answer)
    _add_run_dir(answer, Path("runs/interview-v0"))
    answer.add_argument("--panel", choices=("nemotron", "twin2k"), default="nemotron")
    answer.add_argument("--workers", type=int, default=4)
    answer.add_argument("--full", action="store_true")
    answer.add_argument("--n", type=int, default=50)
    answer.add_argument("--seed", type=int, default=0)
    answer.set_defaults(func=cmd_answer)

    run = sub.add_parser("run", help="build, simulate, and evaluate twin-2k-50 from a config")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument(
        "--run",
        type=Path,
        default=None,
        dest="run_dir",
        help="output directory (default: runs/twin2k50-<panel>-n<n>-seed<seed>)",
    )
    run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (
        ConfigError,
        DatasetError,
        PersonaError,
        InterviewError,
        VisitorError,
        FileNotFoundError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def cmd_fetch(args: argparse.Namespace) -> int:
    data_dir = _resolve(args.data_dir)
    if not args.visitors_only:
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
    if args.visitors or args.visitors_only:
        paths = fetch_visitors(data_dir)
        print("Visitor sources cached:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    out = build_interview(_resolve(args.data_dir), n=args.n, seed=args.seed)
    print(f"Wrote twin-2k-50 interview records to {out}")
    return 0


def cmd_build_visitors(args: argparse.Namespace) -> int:
    out = build_visitors(_resolve(args.data_dir), n=args.n, seed=args.seed)
    print(f"Wrote visitor DAO records to {out}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    data_dir = _resolve(args.data_dir)
    if args.panel == "visitors":
        run_dir = _visitor_run_dir(args)
        run_visitors(
            data_dir,
            run_dir,
            n=args.n,
            seed=args.seed,
            workers=args.workers,
            rebuild=args.rebuild,
        )
        print(f"Wrote visitor visit answers to {run_dir / 'responses.jsonl'}")
        print(f"Wrote visit report to {run_dir / 'report.md'}")
        return 0
    run_dir = _run_dir(args)
    if args.full:
        _simulate_full(data_dir, run_dir, n=args.n, seed=args.seed, workers=args.workers)
    else:
        run_interview(data_dir, run_dir, panel=args.panel, workers=args.workers)
    print(f"Wrote model answers to {run_dir / 'responses.jsonl'}")
    return 0


def cmd_run_visitors(args: argparse.Namespace) -> int:
    data_dir = _resolve(args.data_dir)
    fetch_visitors(data_dir)
    build_visitors(data_dir, n=args.n, seed=args.seed)
    if args.run_dir is None:
        run_dir = Path.cwd() / "runs" / f"visitors-n{args.n}-seed{args.seed}"
    else:
        run_dir = _resolve(args.run_dir)
    print(
        "ACS/ATUS visitors "
        f"n_people={args.n} seed={args.seed} "
        f"turns=8 instrument=acs_atus_visit_v0"
    )
    run_visitors(
        data_dir,
        run_dir,
        n=args.n,
        seed=args.seed,
        workers=args.workers,
        rebuild=False,
    )
    print(f"Wrote visitor visit answers to {run_dir / 'responses.jsonl'}")
    print(f"Wrote visit report to {run_dir / 'report.md'}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    data_dir = _resolve(args.data_dir)
    run_dir = _run_dir(args)
    if args.full:
        evaluate_survey(data_dir, run_dir)
    else:
        evaluate_interview(data_dir, run_dir, panel=args.panel)
    print(f"Wrote the comparison to {run_dir / 'report.md'}")
    return 0


def cmd_answer(args: argparse.Namespace) -> int:
    data_dir = _resolve(args.data_dir)
    run_dir = _run_dir(args)
    if args.full:
        _simulate_full(data_dir, run_dir, n=args.n, seed=args.seed, workers=args.workers)
        evaluate_survey(data_dir, run_dir)
    else:
        run_interview(data_dir, run_dir, panel=args.panel, workers=args.workers)
        evaluate_interview(data_dir, run_dir, panel=args.panel)
    print(f"Wrote the comparison to {run_dir / 'report.md'}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    n = config.effective_n_people
    panel = "nemotron" if config.persona_condition == "nemotron_usa" else "twin2k"
    build_interview(config.data_dir, n=n, seed=config.seed)
    if args.run_dir is None:
        run_dir = Path.cwd() / "runs" / f"twin2k50-{panel}-n{n}-seed{config.seed}"
    else:
        run_dir = _resolve(args.run_dir)
    print(
        "Twin-2K-50 "
        f"panel={panel} "
        f"persona={config.persona_condition} "
        f"n_people={n} "
        f"seed={config.seed}"
    )
    run_interview(config.data_dir, run_dir, panel=panel, workers=args.workers)
    evaluate_interview(config.data_dir, run_dir, panel=panel)
    print(f"Wrote the comparison to {run_dir / 'report.md'}")
    return 0


def _simulate_full(data_dir: Path, run_dir: Path, *, n: int, seed: int, workers: int) -> None:
    from popbench.dao.survey import prepare_survey

    prepared = prepare_survey(data_dir, n_people=n, seed=seed)
    print(
        f"asking {len(prepared['items'])} of {prepared['n_closed']} closed items",
        flush=True,
    )
    answer_survey(prepared["records"], run_dir, workers=workers)


def _add_data_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=Path("data"))


def _add_run_dir(parser: argparse.ArgumentParser, default: Path) -> None:
    parser.add_argument("--run", type=Path, default=default, dest="run_dir")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _run_dir(args: argparse.Namespace) -> Path:
    if getattr(args, "full", False) and args.run_dir == Path("runs/interview-v0"):
        return Path.cwd() / "runs" / "interview-full"
    return _resolve(args.run_dir)


def _visitor_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir == Path("runs/interview-v0"):
        return Path.cwd() / "runs" / f"visitors-n{args.n}-seed{args.seed}"
    return _resolve(args.run_dir)
