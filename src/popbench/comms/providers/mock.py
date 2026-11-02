"""Deterministic provider for tests and keyless dry runs."""

from __future__ import annotations

import json


class MockProvider:
    name = "mock"

    def __init__(self, *, fail_times: int = 0) -> None:
        self.fail_times = fail_times
        self.calls = 0

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float | None = None,
    ) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            from popbench.comms.errors import CommsTransportError

            raise CommsTransportError("mock transport failure")
        system = messages[0]["content"] if messages else ""
        user = messages[-1]["content"] if messages else ""
        if '"answers"' in system:
            return _batch_reply(user, model=model, distribution="distribution" in system)
        if "Do not invent an individual respondent" in system:
            options = _options(user)
            if not options:
                options = ("A", "B")
            weight = 1 / len(options)
            distribution = {option: weight for option in options}
            return json.dumps({"distribution": distribution}, ensure_ascii=False)
        return json.dumps(
            {"answer": "mock", "rationale": f"mock reply from {model}"},
            ensure_ascii=False,
        )


def _batch_reply(user: str, *, model: str, distribution: bool) -> str:
    answers = []
    for question_id, options in _question_blocks(user):
        if distribution:
            labels = options or ("A", "B")
            weight = 1 / len(labels)
            answers.append(
                {
                    "question_id": question_id,
                    "distribution": {label: weight for label in labels},
                    "rationale": f"mock reply from {model}",
                }
            )
        else:
            answers.append(
                {
                    "question_id": question_id,
                    "answer": "mock",
                    "rationale": f"mock reply from {model}",
                }
            )
    return json.dumps({"answers": answers}, ensure_ascii=False)


def _question_blocks(user: str) -> list[tuple[str, tuple[str, ...]]]:
    blocks: list[tuple[str, tuple[str, ...]]] = []
    current: str | None = None
    options: list[str] = []
    collecting = False
    for line in user.splitlines():
        if line.startswith("question_id:"):
            if current is not None:
                blocks.append((current, tuple(options)))
            current = line.split(":", 1)[1].strip()
            options = []
            collecting = False
            continue
        if current is None:
            continue
        if line.strip() == "Options:":
            collecting = True
            continue
        if collecting and line.startswith("- "):
            options.append(line[2:].strip())
        elif collecting and line.strip():
            collecting = False
    if current is not None:
        blocks.append((current, tuple(options)))
    return blocks


def _options(user: str) -> tuple[str, ...]:
    lines = []
    collecting = False
    for line in user.splitlines():
        if line.strip() == "Options:":
            collecting = True
            continue
        if collecting:
            if line.startswith("- "):
                lines.append(line[2:].strip())
            elif line.strip():
                break
    return tuple(lines)
