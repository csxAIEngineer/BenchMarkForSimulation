"""Pack one persona and many questions into a single model request.

Persona methods do not call this. The pipeline groups packages; the client
sends one request per batch and splits the JSON back into per-question rows.
"""

from __future__ import annotations

from collections.abc import Sequence

from popbench.comms.errors import CommsError
from popbench.schema import PromptPackage

ANSWER_BATCH_SYSTEM = (
    "You answer as the person described in the persona. "
    "Stay consistent with that description. "
    "Reply with JSON only, using this schema and no other text: "
    '{"answers": [{"question_id": "...", "answer": "...", "rationale": "..."}]}. '
    "Include one object for every question_id."
)

DISTRIBUTION_BATCH_SYSTEM = (
    "You simulate the response distribution of the described group. "
    "Do not invent an individual respondent. "
    "Reply with JSON only, using this schema and no other text: "
    '{"answers": [{"question_id": "...", "distribution": {"option": 0.5}, "rationale": "..."}]}. '
    "Weights are non-negative and sum to 1. Include one object for every question_id."
)


def pack_batch(packages: Sequence[PromptPackage]) -> tuple[str, str, tuple[str, ...]]:
    """Return system text, user text, and question ids for one subject."""
    if not packages:
        raise CommsError("cannot pack an empty question batch")
    first = packages[0]
    question_ids: list[str] = []
    blocks: list[str] = []
    for package in packages:
        if package.subject_id != first.subject_id or package.method != first.method:
            raise CommsError("a batch must be one subject and one persona method")
        if package.response_mode != first.response_mode or package.persona != first.persona:
            raise CommsError("a batch must share one persona and one response mode")
        if package.question_id in question_ids:
            raise CommsError(f"duplicate question_id in batch: {package.question_id}")
        question_ids.append(package.question_id)
        body = package.question_body or package.user
        blocks.append(f"question_id: {package.question_id}\n{body}")
    system = ANSWER_BATCH_SYSTEM if first.response_mode == "answer" else DISTRIBUTION_BATCH_SYSTEM
    user = "\n\n".join(
        [
            "Persona:",
            first.persona,
            "Questions:",
            "\n\n".join(blocks),
        ]
    )
    return system, user, tuple(question_ids)


def messages_for_batch(packages: Sequence[PromptPackage]) -> list[dict[str, str]]:
    system, user, _ids = pack_batch(packages)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def per_question_payloads(
    parsed: dict | None,
    question_ids: Sequence[str],
    *,
    response_mode: str,
    parse_error: str | None,
) -> list[tuple[dict | None, str | None]]:
    """Map a batch JSON object onto one payload per requested question."""
    if parse_error or parsed is None:
        error = parse_error or "batch JSON did not parse"
        return [(None, error) for _ in question_ids]
    answers = parsed.get("answers")
    if not isinstance(answers, list):
        return [(None, "batch JSON is missing answers") for _ in question_ids]
    by_id: dict[str, dict] = {}
    for item in answers:
        if isinstance(item, dict) and item.get("question_id") not in (None, ""):
            by_id[str(item["question_id"])] = item
    rows: list[tuple[dict | None, str | None]] = []
    for question_id in question_ids:
        item = by_id.get(question_id)
        if item is None:
            rows.append((None, f"missing question_id {question_id}"))
            continue
        rows.append(_one_answer(item, question_id, response_mode))
    return rows


def _one_answer(item: dict, question_id: str, response_mode: str) -> tuple[dict | None, str | None]:
    rationale = item.get("rationale")
    if response_mode == "distribution":
        distribution = item.get("distribution")
        if not isinstance(distribution, dict):
            return None, f"{question_id} is missing distribution"
        return {"distribution": distribution, "rationale": rationale}, None
    if "answer" not in item:
        return None, f"{question_id} is missing answer"
    return {"answer": item.get("answer"), "rationale": rationale}, None
