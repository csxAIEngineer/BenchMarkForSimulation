"""Command line for the population-simulation benchmark.

The main path is five steps: fetch ATUS and Twin-2K, split populations and
question categories, reuse written adult persona cards, answer each paper
category, and score against Twin-2K shares. Visitor interviews, twin-2k-50,
and the six-item probe stay available after those commands.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from popbench import __version__
from popbench.config import ConfigError, load_config
from popbench.dao.atus_categories import (
    build_atus_category_panels,
    category_families,
    category_records,
)
from popbench.dao.atus_populations import (
    POPULATION_FACTORS,
    build_population_visitors,
    split_atus_populations,
)
from popbench.dao.atus_twin2k import build_atus_twin2k, load_all_adult_personas
from popbench.dao.interview import InterviewError, build_interview
from popbench.dao.io import write_jsonl
from popbench.dao.personas import NEMOTRON_SHARD_COUNT, PersonaError, cache_nemotron_shards
from popbench.dao.survey import PAPER_FAMILIES, build_closed_items, items_by_family, write_category_splits
from popbench.dao.twin2k import DatasetError, load_twin2k, twin2k_dir
from popbench.dao.visitors import VisitorError, build_visitors, fetch_visitors, visitors_built_dir
from popbench.evaluate.atus_twin2k import evaluate_atus_twin2k, write_atus_twin2k_comparison
from popbench.evaluate.categories import evaluate_category_run, write_category_comparison
from popbench.evaluate.significance import render_personality_age_report, write_personality_age_evaluation
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
from popbench.simulate.recent import (
    attach_education_recent,
    education_recent_payload,
    education_recent_report,
)
from popbench.simulate.survey import answer_survey, share_blank_instrument
from popbench.simulate.visitors import run_visitors, simulate_population_personas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="popbench",
        description=(
            "Score LLMs on written ATUS persona cards. Fetch the surveys, "
            "split populations and Twin-2K categories, then compare-categories "
            "against Twin-2K shares."
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

    split_populations = sub.add_parser(
        "split-populations",
        help="split ATUS adults into population groups that differ in time use",
    )
    _add_data_dir(split_populations)
    split_populations.set_defaults(func=cmd_split_populations)

    build_population_visitors_cmd = sub.add_parser(
        "build-population-visitors",
        help="write a persona card for every adult in each ATUS population group",
    )
    _add_data_dir(build_population_visitors_cmd)
    build_population_visitors_cmd.set_defaults(func=cmd_build_population_visitors)

    split_categories = sub.add_parser(
        "split-categories",
        help="split Twin-2K responses into one folder per paper category",
    )
    _add_data_dir(split_categories)
    split_categories.set_defaults(func=cmd_split_categories)

    simulate_population_personas_cmd = sub.add_parser(
        "simulate-population-personas",
        help="write one life history per ATUS adult via the DeepSeek API",
    )
    _add_data_dir(simulate_population_personas_cmd)
    simulate_population_personas_cmd.set_defaults(func=cmd_simulate_population_personas)

    answer_population = sub.add_parser(
        "answer-population",
        help="every written ATUS adult card answers Twin-2K closed items; saves under data/twin2k-sim",
    )
    _add_data_dir(answer_population)
    answer_population.add_argument("--workers", type=int, default=8)
    answer_population.add_argument(
        "--families",
        nargs="+",
        choices=PAPER_FAMILIES,
        default=None,
        help="paper categories to answer (default: all seven)",
    )
    answer_population.set_defaults(func=cmd_answer_population)

    compare_categories = sub.add_parser(
        "compare-categories",
        help="answer each Twin-2K paper category using written ATUS adult persona cards",
    )
    _add_data_dir(compare_categories)
    compare_categories.add_argument(
        "--models",
        nargs="+",
        default=list(COMPARE_MODELS),
        help="models to compare (default: gpt-5.5 glm-5.3 deepseek-v4-pro)",
    )
    compare_categories.add_argument("--n", type=int, default=50)
    draw = compare_categories.add_mutually_exclusive_group()
    draw.add_argument(
        "--per-age-band",
        type=int,
        default=None,
        dest="per_age_band",
        help=(
            "sample this many ATUS adults from each age band "
            "(18-29, 30-49, 50-64, 65+), weighted by TUFINLWGT; replaces --n"
        ),
    )
    draw.add_argument(
        "--population",
        choices=tuple(factor.key for factor in POPULATION_FACTORS),
        default=None,
        help=(
            "sample from data/atus/by_population/<factor> level tables, "
            "allocating --n by TUFINLWGT"
        ),
    )
    compare_categories.add_argument("--seed", type=int, default=0)
    compare_categories.add_argument("--workers", type=int, default=4)
    compare_categories.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="sampling temperature sent with each model request (default: omit the field)",
    )
    compare_categories.add_argument(
        "--recent",
        action="store_true",
        help=(
            "append one recent event per person, drawn from the person id, "
            "and write recent.json in the run directory"
        ),
    )
    compare_categories.add_argument(
        "--batch",
        action="store_true",
        help=(
            "ask a category's blank-instrument questions in one request per person; "
            "named instruments stay separate"
        ),
    )
    compare_categories.add_argument(
        "--families",
        nargs="+",
        choices=PAPER_FAMILIES,
        default=None,
        help="paper categories to run (default: all seven)",
    )
    compare_categories.add_argument(
        "--score-only",
        action="store_true",
        help="score responses already saved under the compare directory",
    )
    compare_categories.add_argument(
        "--run",
        type=Path,
        default=None,
        dest="run_dir",
        help=(
            "output directory (default: runs/compare/atus-categories-n<n>-seed<seed>, "
            "atus-categories-age-n<k>-seed<seed> with --per-age-band, "
            "or atus-categories-<factor>-n<n>-seed<seed> with --population; "
            "--temperature, --recent, and --batch append -t<value>, -recent, and -batch)"
        ),
    )
    compare_categories.set_defaults(func=cmd_compare_categories)

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
        help="answer six Twin-2K items using written ATUS adult persona cards",
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


def cmd_split_categories(args: argparse.Namespace) -> int:
    data_dir = _resolve(args.data_dir)
    data = load_twin2k(data_dir, download=False)
    dest = twin2k_dir(data_dir) / "by_category"
    summary = write_category_splits(data, dest)
    print(f"Wrote {len(summary['categories'])} Twin-2K categories to {dest}")
    for row in summary["categories"]:
        wave4 = f", wave4 columns {row['wave4_columns']}" if row["wave4_columns"] else ""
        print(
            f"  {row['dir']} ({row['label']}): "
            f"{row['n_questions']} catalog entries, "
            f"wave1_3 columns {row['wave1_3_columns']}{wave4}"
        )
    return 0


def cmd_split_populations(args: argparse.Namespace) -> int:
    data_dir = _resolve(args.data_dir)
    summary = split_atus_populations(data_dir)
    kept = [row for row in summary["factors"] if row["kept"]]
    print(
        f"Wrote {len(kept)} ATUS population factors "
        f"({summary['n_adults']} adults) to {summary['dir']}"
    )
    for row in summary["factors"]:
        groups = ", ".join(f"{group['level']} {group['n']}" for group in row["groups"])
        status = "kept" if row["kept"] else "dropped"
        print(f"  {row['factor']} ({row['label']}, {status}): joint p={_format_p(row['p_value'])}; {groups}")
        if row["levels_omitted"]:
            omitted = ", ".join(
                f"{item['level']} {item['n']}" for item in row["levels_omitted"]
            )
            print(f"    omitted: {omitted}")
        if row["kept"]:
            named = [item["activity"] for item in row["activities"] if item["significant"]]
            print(f"    significant activities: {', '.join(named)}")
    return 0


def _format_p(p_value: float | None) -> str:
    if p_value is None:
        return "n/a"
    if p_value == 0:
        return "<1e-300"
    return f"{p_value:.3g}"


def cmd_build(args: argparse.Namespace) -> int:
    out = build_interview(_resolve(args.data_dir), n=args.n, seed=args.seed)
    print(f"Wrote twin-2k-50 interview records to {out}")
    return 0


def cmd_build_visitors(args: argparse.Namespace) -> int:
    out = build_visitors(_resolve(args.data_dir), n=args.n, seed=args.seed)
    print(f"Wrote visitor DAO records to {out}")
    return 0


def cmd_build_population_visitors(args: argparse.Namespace) -> int:
    summary = build_population_visitors(_resolve(args.data_dir))
    print(
        f"Wrote {summary['n_cards']} ATUS population persona cards "
        f"({summary['n_groups']} groups) to {summary['dir']}"
    )
    for group in summary["groups"]:
        print(f"  {group['factor']}/{group['level']}: {group['n_people']}")
    return 0


def cmd_answer_population(args: argparse.Namespace) -> int:
    """Ask every written adult card the closed Twin-2K items. Saves under data/twin2k-sim."""
    data_dir = _resolve(args.data_dir)
    settings = model_settings_from_env()
    personas = load_all_adult_personas(data_dir)
    dataset = load_twin2k(data_dir, download=False)
    grouped = items_by_family(build_closed_items(dataset.catalog))
    families = category_families(list(args.families) if args.families else None)
    root = data_dir / "twin2k-sim" / run_slug(settings.model)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": settings.model,
        "base_url": settings.base_url,
        "transport": settings.transport,
        "n_people": len(personas),
        "source": "visitors/by_population/adults/personas.jsonl",
        "questions": "Twin-2K closed items. Each paper category is a separate interview.",
        "families": families,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(
        f"answer all adult cards model={settings.model} n={len(personas)} "
        f"families={','.join(families)} -> {root}",
        flush=True,
    )
    for family in families:
        items = grouped[family]
        if not items:
            raise InterviewError(f"{family} has no closed items")
        records = category_records(personas, items)
        run_dir = root / family
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "items.json").write_text(json.dumps(items, ensure_ascii=False) + "\n")
        print(
            f"simulating {settings.model} {family} "
            f"({len(items)} items, {len(records)} people) -> {run_dir}",
            flush=True,
        )
        answer_survey(
            records,
            run_dir,
            workers=args.workers,
            settings=settings,
            panel="atus-all-adults",
            family=family,
            resume=True,
        )
        print(f"saved {family} to {run_dir / 'responses.jsonl'}", flush=True)
    print(f"Wrote population answers to {root}")
    return 0


def cmd_simulate_population_personas(args: argparse.Namespace) -> int:
    summary = simulate_population_personas(_resolve(args.data_dir))
    print(
        f"Wrote {summary['n_cards']} adult persona cards with {summary['model']}, "
        f"reused across {summary['n_group_rows']} group rows "
        f"({summary['n_groups']} groups) to {summary['dir']}"
    )
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


def cmd_compare_categories(args: argparse.Namespace) -> int:
    """Answer each paper category from written ATUS adult cards, then score by category."""
    data_dir = _resolve(args.data_dir)
    root = _categories_root(args)
    families = category_families(list(args.families) if args.families else None)
    minimum = 1 if args.per_age_band else 2
    slugs = _compare_slugs(args.models, minimum=minimum)
    settings_list = None if args.score_only else _compare_settings(args.models, minimum=minimum)
    built = build_atus_category_panels(
        data_dir,
        n=args.n,
        seed=args.seed,
        out_dir=root,
        families=families,
        n_per_age_band=args.per_age_band,
        population=args.population,
    )
    if args.population:
        sampling = f"population={args.population} n={args.n}"
    elif args.per_age_band:
        sampling = f"per_age_band={args.per_age_band} n={built['n']}"
    else:
        sampling = f"n={args.n}"
    temperature = getattr(args, "temperature", None)
    recent_summary = None
    if args.recent:
        payload = education_recent_payload(built["personas"])
        (root / "recent.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        recent_summary = education_recent_report(payload)
        for panel in built["panels"]:
            panel["records"] = [attach_education_recent(record) for record in panel["records"]]
    if args.batch:
        for panel in built["panels"]:
            panel["records"] = share_blank_instrument(panel["records"], panel["family"])
    details = sampling
    if temperature is not None:
        details += f" temperature={format(temperature, 'g')}"
    if args.recent:
        details += " recent=1"
    if args.batch:
        details += " batch=1"
    print(
        "compare atus personas by twin-2k category "
        f"models={','.join(slugs)} {details} seed={args.seed} "
        f"families={','.join(panel['family'] for panel in built['panels'])}",
        flush=True,
    )
    summaries = []
    for index, slug in enumerate(slugs):
        for panel in built["panels"]:
            run_dir = root / slug / panel["family"]
            if args.score_only:
                print(f"scoring {slug} {panel['family']}", flush=True)
            else:
                settings = settings_list[index]
                if temperature is not None:
                    settings = replace(settings, temperature=temperature)
                print(
                    f"simulating {settings.model} {panel['family']} "
                    f"({len(panel['items'])} items) -> {run_dir}",
                    flush=True,
                )
                answer_survey(
                    panel["records"],
                    run_dir,
                    workers=args.workers,
                    settings=settings,
                    panel="atus",
                    seed=args.seed,
                    family=panel["family"],
                    batch=args.batch,
                )
            summary = evaluate_category_run(
                run_dir,
                panel["items"],
                panel["baseline"],
                family=panel["family"],
                seed=args.seed,
            )
            summaries.append(summary)
            print(
                f"  {summary['model']} {summary['label']} "
                f"tvd={summary['mean_tvd']} mae={summary['mean_mae']}",
                flush=True,
            )
            if args.per_age_band and panel["family"] == "personality":
                print(f"testing {summary['model']} personality against Twin-2K", flush=True)
                significance = write_personality_age_evaluation(
                    run_dir,
                    built["personas"],
                    panel["items"],
                    data_dir,
                    model=summary["model"],
                    seed=args.seed,
                    n_per_age_band=args.per_age_band,
                )
                print(
                    f"  gof_holm={significance['overall']['n_gof_holm']}/"
                    f"{significance['overall']['n_tested']} "
                    f"gradient_r={significance['age_effects']['gradient_r']}",
                    flush=True,
                )
                if len(slugs) == 1:
                    (root / "compare.md").write_text(render_personality_age_report(significance))
                    (root / "significance.json").write_text(json.dumps(significance) + "\n")
    if len(slugs) >= 2:
        report = write_category_comparison(
            root,
            summaries,
            population=built.get("population"),
            temperature=temperature,
            recent=recent_summary,
            batch=args.batch,
        )
        print("per category (lower Twin-2K total variation distance ranks first)")
        for category in report["categories"]:
            winner = category["ranked"][0]
            print(f"  {category['label']}: {winner['model']} tvd={winner['mean_tvd']}")
    print(f"Wrote category comparison to {root / 'compare.md'}")
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


def _compare_slugs(names: list[str], *, minimum: int = 2) -> list[str]:
    slugs: list[str] = []
    for name in names:
        slug = run_slug(name)
        if slug in slugs:
            raise SettingsError(f"duplicate model: {slug}")
        slugs.append(slug)
    if len(slugs) < minimum:
        if minimum > 1:
            raise CompareError("compare needs at least two models")
        raise CompareError("compare needs a model")
    return slugs


def _compare_settings(names: list[str], *, minimum: int = 2) -> list[ModelSettings]:
    slugs = _compare_slugs(names, minimum=minimum)
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


def _categories_root(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        return _resolve(args.run_dir)
    population = getattr(args, "population", None)
    if population:
        name = f"atus-categories-{population}-n{args.n}-seed{args.seed}"
    elif getattr(args, "per_age_band", None):
        name = f"atus-categories-age-n{args.per_age_band}-seed{args.seed}"
    else:
        name = f"atus-categories-n{args.n}-seed{args.seed}"
    return Path.cwd() / "runs" / "compare" / f"{name}{_categories_suffix(args)}"


def _categories_suffix(args: argparse.Namespace) -> str:
    """Keep a temperature or recent run out of the directory that omits them."""
    parts: list[str] = []
    temperature = getattr(args, "temperature", None)
    if temperature is not None:
        parts.append("t" + format(float(temperature), "g"))
    if getattr(args, "recent", False):
        parts.append("recent")
    if getattr(args, "batch", False):
        parts.append("batch")
    if not parts:
        return ""
    return "-" + "-".join(parts)


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
