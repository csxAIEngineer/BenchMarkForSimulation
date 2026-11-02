"""Command line for fetch and a minimal dataset → persona → LLM path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from popbench import __version__
from popbench.pipeline import prepare_bundle, simulate, write_results
from popbench.comms import LLMClient, build_provider
from popbench.comms.errors import CommsError
from popbench.datasets import DATASETS, resolve_dataset
from popbench.datasets.io import DatasetError
from popbench.models import METHODS
from popbench.models.errors import PersonaMethodError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description="三层人口模拟基准：datasets 取数，models 打包人物卡，comms 调用 LLM。",
    )
    parser.add_argument("--version", action="version", version=f"benchmark {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="下载并缓存一个或全部数据集")
    fetch.add_argument("dataset", choices=[*DATASETS, "all"])
    fetch.add_argument("--data-dir", type=Path, default=Path("data"))
    fetch.add_argument("--limit", type=int, default=None, help="每个 split 最多缓存多少条，便于试拉")
    fetch.add_argument("--study", default=None, help="Mega-Study 的 study 名；默认全部 19 个")
    fetch.add_argument("--split", default=None, help="SimBench 的 SimBenchPop 或 SimBenchGrouped")
    fetch.set_defaults(func=cmd_fetch)

    run = sub.add_parser("simulate", help="用一种人物卡方法走通最小问答路径")
    run.add_argument("--dataset", required=True)
    run.add_argument("--method", default="demographics_only", choices=sorted(METHODS))
    run.add_argument("--provider", default="mock", choices=("mock", "openai", "cr"))
    run.add_argument("--model", default=None, help="provider 上的模型名；mock 默认为 mock-llm")
    run.add_argument("--data-dir", type=Path, default=Path("data"))
    run.add_argument("--limit", type=int, default=1)
    run.add_argument(
        "--batch-size",
        type=int,
        default=15,
        help="同一人物卡下每次请求合并的题数（默认 15）",
    )
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--study", default=None)
    run.add_argument("--split", default=None)
    run.add_argument(
        "--attach-twin2k",
        action="store_true",
        help="Mega-Study 按 pid 接上已缓存的 Twin-2K-500 人物字段",
    )
    run.add_argument("--out", type=Path, default=Path("runs/minimal"))
    run.add_argument("--temperature", type=float, default=0)
    run.set_defaults(func=cmd_simulate)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (DatasetError, PersonaMethodError, CommsError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def cmd_fetch(args: argparse.Namespace) -> int:
    names = list(DATASETS) if args.dataset == "all" else [resolve_dataset(args.dataset)]
    data_dir = _resolve(args.data_dir)
    for name in names:
        loader = DATASETS[name]
        kwargs = {"limit": args.limit}
        if name == "twin2k_mega_study":
            kwargs["study"] = args.study
        if name == "simbench":
            kwargs["split"] = args.split
        path = loader.download(data_dir, **kwargs)
        bundle = loader.load(data_dir, study=args.study, split=args.split)
        print(
            f"{loader.display_name}: subjects={len(bundle.subjects)} "
            f"cases={len(bundle.cases)} cache={path}"
        )
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    name = resolve_dataset(args.dataset)
    loader = DATASETS[name]
    data_dir = _resolve(args.data_dir)
    bundle = loader.load(data_dir, limit=None, study=args.study, split=args.split)
    twin = None
    if name == "twin2k_mega_study" and args.method != "group_prompt":
        if not args.attach_twin2k:
            raise PersonaMethodError(
                "Mega-Study 行里没有人物卡。请先 benchmark fetch twin2k_500，"
                "再加 --attach-twin2k。"
            )
        twin = DATASETS["twin2k_500"].load(data_dir)
    prepared = prepare_bundle(bundle, method=args.method, twin=twin)
    model = args.model or ("mock-llm" if args.provider == "mock" else _default_model(args.provider))
    provider = build_provider(args.provider)
    cache_dir = _resolve(args.out) / "cache"
    client = LLMClient(
        provider,
        model=model,
        cache_dir=cache_dir,
        temperature=None if args.provider == "mock" else args.temperature,
    )
    results = simulate(
        prepared,
        method=args.method,
        client=client,
        limit=args.limit,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    out = _resolve(args.out)
    write_results(
        out,
        results,
        manifest={
            "dataset": name,
            "method": args.method,
            "provider": args.provider,
            "model": model,
            "n": len(results),
            "batch_size": args.batch_size,
        },
    )
    print(f"模拟 {len(results)} 条，方法 {args.method}，provider {args.provider} -> {out / 'responses.jsonl'}")
    for result in results:
        answer = None if result.parsed is None else result.parsed
        print(f"  {result.subject_id} {result.question_id} parsed={answer}")
    return 0


def _default_model(provider: str) -> str:
    if provider == "cr":
        return "deepseek-v4-pro"
    return "gpt-4.1-mini"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


if __name__ == "__main__":
    raise SystemExit(main())
