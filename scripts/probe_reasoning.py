#!/usr/bin/env python3
"""One short chat per model and reasoning_effort value. Not a Twin sample."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cr_api import CRClient, CRError  # noqa: E402

CANDIDATES = (None, "none", "low", "medium", "high")
PROMPT = [{"role": "user", "content": "Reply with the single word ok."}]


def _probe(client: CRClient, model: str, effort: str | None, temperature: float, timeout: float) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "messages": PROMPT,
        "model": model,
        "temperature": temperature,
        "timeout": timeout,
    }
    if effort is not None:
        kwargs["reasoning_effort"] = effort
    try:
        result = client.chat(**kwargs)
    except CRError as exc:
        return {
            "accepted": False,
            "error": str(exc)[:500],
            "reasoning_content_nonempty": False,
            "reasoning_effort": "default-omitted" if effort is None else effort,
            "status": getattr(exc, "status", None),
            "temperature": temperature,
        }
    reasoning = str(getattr(result, "reasoning_content", "") or "")
    return {
        "accepted": True,
        "content_chars": len(str(result.content or "")),
        "error": None,
        "reasoning_content_chars": len(reasoning),
        "reasoning_content_nonempty": bool(reasoning.strip()),
        "reasoning_effort": "default-omitted" if effort is None else effort,
        "status": 200,
        "temperature": temperature,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Probe CR reasoning_effort values. One short chat each.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    client = CRClient()
    rows = []
    for model in args.models:
        for effort in CANDIDATES:
            row = {"model": model, **_probe(client, model, effort, args.temperature, args.timeout)}
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("model", "reasoning_effort", "accepted", "status", "reasoning_content_nonempty", "error")}, ensure_ascii=False), flush=True)
    Path(args.output).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
