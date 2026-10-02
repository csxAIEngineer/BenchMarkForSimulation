"""随机抽一条真实 ACS/ATUS 访问者，生成即将发给大模型的完整提示文本。

用法（项目根目录）::

    python tests/preview_llm_prompt.py
    python tests/preview_llm_prompt.py --seed 42 --out /tmp/prompt.txt
    python tests/preview_llm_prompt.py --data-dir data --n-panel 50

不调用 API，只预览 system + 首轮 user。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from popbench.dao.visitors import (  # noqa: E402
    VisitorError,
    build_visitors,
    cache_is_complete,
    fetch_visitors,
    load_visitor_panel,
    visitors_built_dir,
)
from popbench.simulate.persona_background import expand_persona  # noqa: E402
from popbench.simulate.visitors import (  # noqa: E402
    VISIT_TURNS,
    system_message,
    visit_user_message,
)


def ensure_visitor_panel(data_dir: Path, n: int, seed: int) -> Path:
    """Ensure a real visitor panel exists; fetch/build if needed."""
    if not cache_is_complete(data_dir):
        fetch_visitors(data_dir)
    path = visitors_built_dir(data_dir) / f"visitors_n{n}.jsonl"
    if not path.is_file():
        build_visitors(data_dir, n=n, seed=seed)
    if not path.is_file():
        raise VisitorError(f"visitor panel missing after build: {path}")
    return path


def pick_visitor(
    rows: list[dict[str, Any]], seed: int | None
) -> tuple[dict[str, Any], int]:
    """Randomly select one real visitor row."""
    if not rows:
        raise VisitorError("visitor panel is empty")
    rng = np.random.default_rng(seed)
    index = int(rng.integers(0, len(rows)))
    return rows[index], index


def build_llm_payload(
    row: dict[str, Any],
    *,
    seed: int,
    turn_index: int = 0,
) -> dict[str, Any]:
    """Build the messages that simulate would send to the model."""
    visitor_id = str(row["id"])
    demographics = dict(row.get("demographics") or {})
    if row.get("state") and "state" not in demographics:
        demographics["state"] = row["state"]
    expanded = expand_persona(
        visitor_id=visitor_id,
        demographics=demographics,
        routine=row.get("routine") or {},
        base_text=str(row.get("persona", {}).get("text") or ""),
        seed=seed,
    )
    turn = VISIT_TURNS[turn_index]
    system = system_message(expanded["persona_text"])
    user = visit_user_message(turn)
    return {
        "visitor_id": visitor_id,
        "state": row.get("state"),
        "demographics": demographics,
        "background": expanded["background"],
        "turn": {"item_id": turn.item_id, "turn": turn.turn, "prompt": turn.prompt},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "preview_text": (
            "===== SYSTEM =====\n"
            f"{system}\n\n"
            "===== USER (turn 1) =====\n"
            f"{user}\n"
        ),
    }


def preview_random_llm_prompt(
    data_dir: Path,
    *,
    seed: int | None = None,
    n_panel: int = 50,
    panel_seed: int = 0,
) -> dict[str, Any]:
    """Pick one real dataset visitor and return the LLM-bound prompt preview."""
    draw_seed = 0 if seed is None else int(seed)
    path = ensure_visitor_panel(data_dir, n=n_panel, seed=panel_seed)
    rows = load_visitor_panel(path)
    row, index = pick_visitor(rows, seed=draw_seed)
    payload = build_llm_payload(row, seed=panel_seed if seed is None else draw_seed)
    payload["panel_path"] = str(path)
    payload["panel_index"] = index
    payload["panel_size"] = len(rows)
    payload["draw_seed"] = draw_seed
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="随机抽真实访问者，打印将发给大模型的 system/user 文本",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--seed", type=int, default=None, help="抽人与扩写用的种子")
    parser.add_argument("--n-panel", type=int, default=50, help="面板人数（缺失则重建）")
    parser.add_argument("--panel-seed", type=int, default=0, help="建面板用的种子")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="可选：把 preview_text 写到文件",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出完整 JSON（含 messages / background）",
    )
    args = parser.parse_args(argv)

    try:
        payload = preview_random_llm_prompt(
            args.data_dir,
            seed=args.seed,
            n_panel=args.n_panel,
            panel_seed=args.panel_seed,
        )
    except VisitorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    header = (
        f"# visitor={payload['visitor_id']} "
        f"panel_index={payload['panel_index']}/{payload['panel_size']} "
        f"draw_seed={payload['draw_seed']} "
        f"state={payload.get('state')}\n"
    )
    text = header + payload["preview_text"]
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.json:
            args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        else:
            args.out.write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)
    if args.json and not args.out:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
