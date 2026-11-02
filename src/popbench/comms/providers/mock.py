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
        user = messages[-1]["content"] if messages else ""
        if "Do not invent an individual respondent" in messages[0]["content"]:
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
