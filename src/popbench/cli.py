"""Command line for the population-simulation benchmark.

The dataset (people, items, human baseline) stays fixed. `dao` builds that
panel, `simulate` answers it with the model under test, and `evaluate` scores
the saved answers. `run` does those three steps for the twin-2k-50 config.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from popbench import __version__
from popbench.config import ConfigError, load_config
from popbench.dao.atus_twin2k import build_atus_twin2k
from popbench.dao.interview import InterviewError, build_interview
from popbench.dao.io import write_jsonl
from popbench.dao.personas import NEMOTRON_SHARD_COUNT, PersonaError, cache_nemotron_shards
from popbench.dao.twin2k import DatasetError, load_twin2k
from popbench.dao.visitors import VisitorError, build_visitors, fetch_visitors, visitors_built_dir
from popbench.evaluate.atus_twin2k import evaluate_atus_twin2k, write_atus_twin2k_comparison
from popbench.evaluate.compare import CompareError
from popbench.evaluate.interview import evaluate_interview
from popbench.evaluate.robustness import RobustnessError, audit_cross_model_variance
from popbench.evaluate.survey import evaluate_survey
from popbench.evaluate.visitors import VisitorEvalError, evaluate_visitors
from popbench.simulate.client import (
    COMPARE_MODELS,
    ModelSettings,
    SettingsError,
    model_settings_for,
    model_settings_from_env,
    run_slug,
)
from popbench.simulate.interview import run_interview, run_records
from popbench.simulate.survey import answer_survey
from popbench.simulate.visitors import run_visitors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="popbench",
        description=(
            "Compare LLMs on a fixed population dataset: build the panel, "
            "simulate answers, and score them against the human baseline."
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
    _add_model(simulate)
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
    _add_model(run_visitors_cmd)
    run_visitors_cmd.add_argument("--workers", type=int, default=4)
    run_visitors_cmd.add_argument(
        "--run",
        type=Path,
        default=None,
        dest="run_dir",
        help="output directory (default: runs/visitors-n<n>-seed<seed>)",
    )
    run_visitors_cmd.set_defaults(func=cmd_run_visitors)

    evaluate = sub.add_parser(
        "evaluate",
        help="score a saved run (twin-2k-50 or ACS/ATUS visitors)",
    )
    _add_data_dir(evaluate)
    _add_run_dir(evaluate, Path("runs/interview-v0"))
    evaluate.add_argument(
        "--panel",
        choices=("nemotron", "twin2k", "visitors"),
        default=None,
        help="visitors = ATUS behavioral fit; nemotron/twin2k = Twin-2K scores",
    )
    evaluate.add_argument(
        "--full",
        action="store_true",
        help="score the longer closed-ended form",
    )
    evaluate.set_defaults(func=cmd_evaluate)

    score = sub.add_parser("score", help="alias of evaluate")
    _add_data_dir(score)
    _add_run_dir(score, Path("runs/interview-v0"))
    score.add_argument("--panel", choices=("nemotron", "twin2k", "visitors"), default=None)
    score.add_argument("--full", action="store_true")
    score.set_defaults(func=cmd_evaluate)

    audit = sub.add_parser(
        "audit-robustness",
        help="cross-model variance audit over visitor simulate runs",
    )
    audit.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        required=True,
        help="two or more visitor run directories (different models)",
    )
    audit.add_argument(
        "--out",
        type=Path,
        default=None,
        dest="out_dir",
        help="output directory (default: <parent-of-first-run>/robustness)",
    )
    audit.set_defaults(func=cmd_audit_robustness)

    answer = sub.add_parser("answer-interview", help="simulate a panel, then evaluate it")
    _add_data_dir(answer)
    _add_run_dir(answer, Path("runs/interview-v0"))
    answer.add_argument("--panel", choices=("nemotron", "twin2k"), default="nemotron")
    _add_model(answer)
    answer.add_argument("--workers", type=int, default=4)
    answer.add_argument("--full", action="store_true")
    answer.add_argument("--n", type=int, default=50)
    answer.add_argument("--seed", type=int, default=0)
    answer.set_defaults(func=cmd_answer)

    run = sub.add_parser("run", help="build, simulate, and evaluate twin-2k-50 from a config")
    run.add_argument("--config", type=Path, required=True)
    _add_model(run)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument(
        "--run",
        type=Path,
        default=None,
        dest="run_dir",
        help="output directory (default: runs/twin2k50-<panel>-n<n>-seed<seed>)",
    )
    run.set_defaults(func=cmd_run)

    compare = sub.add_parser(
        "compare",
        help="answer Twin-2K items as ATUS personas and rank models against Twin-2K shares",
    )
    _add_data_dir(compare)
    compare.add_argument(
        "--models",
        nargs="+",
        default=list(COMPARE_MODELS),
        help="models to compare (default: gpt-5.5 glm-5.3 deepseek-v4-pro)",
    )
    compare.add_argument("--n", type=int, default=50)
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument("--workers", type=int, default=4)
    compare.add_argument(
        "--score-only",
        action="store_true",
        help="rank runs already saved under the compare directory",
    )
    compare.add_argument(
        "--run",
        type=Path,
        default=None,
        dest="run_dir",
        help="output directory (default: runs/compare/atus-twin2k-n<n>-seed<seed>)",
    )
    compare.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (
        ConfigError,
        DatasetError,
        PersonaError,
        InterviewError,
        VisitorError,
        VisitorEvalError,
        RobustnessError,
        CompareError,
        SettingsError,
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
    settings = _model_settings(args)
    if args.panel == "visitors":
        run_dir = _visitor_run_dir(args)
        run_visitors(
            data_dir,
            run_dir,
            n=args.n,
            seed=args.seed,
            workers=args.workers,
            rebuild=args.rebuild,
            settings=settings,
        )
        print(f"Wrote visitor visit answers to {run_dir / 'responses.jsonl'}")
        print(f"Wrote visit report to {run_dir / 'report.md'}")
        return 0
    run_dir = _run_dir(args)
    if args.full:
        _simulate_full(
            data_dir,
            run_dir,
            n=args.n,
            seed=args.seed,
            workers=args.workers,
            settings=settings,
        )
    else:
        run_interview(
            data_dir,
            run_dir,
            panel=args.panel,
            workers=args.workers,
            settings=settings,
        )
    print(f"Wrote model answers to {run_dir / 'responses.jsonl'}")
    return 0


def cmd_run_visitors(args: argparse.Namespace) -> int:
    data_dir = _resolve(args.data_dir)
    settings = _model_settings(args)
    fetch_visitors(data_dir)
    build_visitors(data_dir, n=args.n, seed=args.seed)
    if args.run_dir is None:
        suffix = f"-{run_slug(args.model)}" if args.model else ""
        run_dir = Path.cwd() / "runs" / f"visitors{suffix}-n{args.n}-seed{args.seed}"
    else:
        run_dir = _resolve(args.run_dir)
    print(
        "ACS/ATUS visitors "
        f"model={settings.model} "
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
        settings=settings,
    )
    print(f"Wrote visitor visit answers to {run_dir / 'responses.jsonl'}")
    print(f"Wrote visit report to {run_dir / 'report.md'}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    data_dir = _resolve(args.data_dir)
    run_dir = _run_dir(args)
    if args.panel == "visitors" or (
        args.panel is None and (run_dir / "option_shares.json").is_file()
        and (run_dir / "meta.json").is_file()
        and json.loads((run_dir / "meta.json").read_text()).get("panel") == "visitors"
    ):
        summary = evaluate_visitors(data_dir, run_dir)
        fit = summary["behavioral_fit"]
        print(
            "visitor fit "
            f"corr={fit.get('distribution_correlation')} "
            f"share_mse={fit.get('share_mse')} "
            f"coverage={fit.get('response_amplitude_coverage')}"
        )
        print(f"Wrote visitor evaluation to {run_dir / 'eval_summary.json'}")
        print(f"Updated {run_dir / 'report.md'}")
        return 0
    if args.full:
        evaluate_survey(data_dir, run_dir)
    else:
        evaluate_interview(data_dir, run_dir, panel=args.panel)
    print(f"Wrote the comparison to {run_dir / 'report.md'}")
    return 0


def cmd_audit_robustness(args: argparse.Namespace) -> int:
    runs = [_resolve(path) for path in args.runs]
    out = None if args.out_dir is None else _resolve(args.out_dir)
    summary = audit_cross_model_variance(runs, out_dir=out)
    stats = summary["mean_code_variance"]
    print(
        "robustness "
        f"models={summary['n_runs']} "
        f"mean_code_var={stats['mean_variance']:.4f} "
        f"mean_code_std={stats['mean_std']:.4f}"
    )
    print(f"Wrote robustness audit to {summary['out_dir']}")
    return 0


def cmd_answer(args: argparse.Namespace) -> int:
    data_dir = _resolve(args.data_dir)
    settings = _model_settings(args)
    run_dir = _run_dir(args)
    if args.full:
        _simulate_full(
            data_dir,
            run_dir,
            n=args.n,
            seed=args.seed,
            workers=args.workers,
            settings=settings,
        )
        evaluate_survey(data_dir, run_dir)
    else:
        run_interview(
            data_dir,
            run_dir,
            panel=args.panel,
            workers=args.workers,
            settings=settings,
        )
        evaluate_interview(data_dir, run_dir, panel=args.panel)
    print(f"Wrote the comparison to {run_dir / 'report.md'}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    settings = _model_settings(args)
    n = config.effective_n_people
    panel = "nemotron" if config.persona_condition == "nemotron_usa" else "twin2k"
    build_interview(config.data_dir, n=n, seed=config.seed)
    if args.run_dir is None:
        suffix = f"-{run_slug(settings.model)}" if args.model else ""
        run_dir = Path.cwd() / "runs" / f"twin2k50-{panel}{suffix}-n{n}-seed{config.seed}"
    else:
        run_dir = _resolve(args.run_dir)
    print(
        "Twin-2K-50 "
        f"model={settings.model} "
        f"panel={panel} "
        f"persona={config.persona_condition} "
        f"n_people={n} "
        f"seed={config.seed}"
    )
    run_interview(
        config.data_dir,
        run_dir,
        panel=panel,
        workers=args.workers,
        settings=settings,
    )
    evaluate_interview(config.data_dir, run_dir, panel=panel)
    print(f"Wrote the comparison to {run_dir / 'report.md'}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Answer a few Twin-2K items as ATUS personas, then rank against real shares."""
    data_dir = _resolve(args.data_dir)
    root = _compare_root(args)
    slugs = _compare_slugs(args.models)
    settings_list = None if args.score_only else _compare_settings(args.models)
    built = build_atus_twin2k(data_dir, n=args.n, seed=args.seed)
    root.mkdir(parents=True, exist_ok=True)
    (root / "items.json").write_text(json.dumps(built["items"], indent=2) + "\n")
    (root / "human_baseline.json").write_text(json.dumps(built["baseline"], indent=2) + "\n")
    write_jsonl(root / "personas.jsonl", built["personas"])
    print(
        "compare atus personas on twin-2k items "
        f"models={','.join(slugs)} n={args.n} seed={args.seed} "
        f"items={len(built['items'])}",
        flush=True,
    )
    summaries = []
    run_dirs = []
    item_ids = [item["item_id"] for item in built["items"]]
    for index, slug in enumerate(slugs):
        run_dir = root / slug
        run_dirs.append(run_dir)
        if args.score_only:
            print(f"scoring {slug}", flush=True)
        else:
            settings = settings_list[index]
            print(f"simulating {settings.model} -> {run_dir}", flush=True)
            run_records(
                built["records"],
                run_dir,
                workers=args.workers,
                settings=settings,
                panel="atus",
                seed=args.seed,
            )
        summary = evaluate_atus_twin2k(
            run_dir,
            built["records"],
            built["baseline"],
            seed=args.seed,
        )
        summaries.append(summary)
        print(
            f"  {summary.get('model') or slug} "
            f"tvd={summary.get('mean_tvd')} "
            f"code_mae={summary.get('mean_abs_code_error')}",
            flush=True,
        )
    report = write_atus_twin2k_comparison(root, summaries)
    audit_cross_model_variance(run_dirs, out_dir=root / "robustness", item_ids=item_ids)
    print("ranking (lower Twin-2K total variation distance, then lower mean-code error)")
    for row in report["ranked"]:
        print(
            f"  {row['place']}. {row['model']} "
            f"tvd={row['mean_tvd']} "
            f"code_mae={row['mean_abs_code_error']}"
        )
    print(f"Wrote model comparison to {root / 'compare.md'}")
    return 0


def _simulate_full(
    data_dir: Path,
    run_dir: Path,
    *,
    n: int,
    seed: int,
    workers: int,
    settings: ModelSettings | None = None,
) -> None:
    from popbench.dao.survey import prepare_survey

    prepared = prepare_survey(data_dir, n_people=n, seed=seed)
    print(
        f"asking {len(prepared['items'])} of {prepared['n_closed']} closed items",
        flush=True,
    )
    answer_survey(prepared["records"], run_dir, workers=workers, settings=settings)


def _add_model(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default=None,
        help="model under test: gpt-5.5, glm-5.3, deepseek-v4-pro via CR, or a DeepSeek model id",
    )


def _model_settings(args: argparse.Namespace) -> ModelSettings:
    if getattr(args, "model", None):
        return model_settings_for(args.model)
    return model_settings_from_env()


def _compare_slugs(names: list[str]) -> list[str]:
    slugs: list[str] = []
    for name in names:
        slug = run_slug(name)
        if slug in slugs:
            raise SettingsError(f"duplicate model: {slug}")
        slugs.append(slug)
    if len(slugs) < 2:
        raise CompareError("compare needs at least two models")
    return slugs


def _compare_settings(names: list[str]) -> list[ModelSettings]:
    slugs = _compare_slugs(names)
    settings: list[ModelSettings] = []
    missing: list[str] = []
    for name in names:
        try:
            settings.append(model_settings_for(name))
        except SettingsError as exc:
            missing.append(str(exc))
    if missing:
        unique = list(dict.fromkeys(missing))
        raise SettingsError("; ".join(unique))
    resolved = [run_slug(item.model) for item in settings]
    if resolved != slugs:
        raise SettingsError("model ids did not match the compare directory names")
    return settings


def _compare_root(args: argparse.Namespace) -> Path:
    if args.run_dir is None:
        return Path.cwd() / "runs" / "compare" / f"atus-twin2k-n{args.n}-seed{args.seed}"
    return _resolve(args.run_dir)


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
        suffix = f"-{run_slug(args.model)}" if getattr(args, "model", None) else ""
        return Path.cwd() / "runs" / f"visitors{suffix}-n{args.n}-seed{args.seed}"
    return _resolve(args.run_dir)
