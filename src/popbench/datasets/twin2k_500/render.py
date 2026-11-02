"""Turn Twin-2K persona and wave-4 JSON into text without copying gold into prompts."""

from __future__ import annotations

from typing import Any

from popbench.datasets.textutil import answer_label, option_labels, strip_html


def elements_of(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        elements = payload.get("Elements", [])
        return [item for item in elements if isinstance(item, dict)] if isinstance(elements, list) else []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def iter_blocks(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for element in elements:
        kind = element.get("ElementType")
        if kind == "Block":
            blocks.append(element)
        elif kind == "Branch":
            nested = element.get("Elements") or []
            if isinstance(nested, list):
                blocks.extend(iter_blocks([item for item in nested if isinstance(item, dict)]))
    return blocks


def demographic_fields(payload: Any) -> tuple[str, dict[str, str]]:
    """First survey block only, matching the Mega-Study demographic variant."""
    blocks = iter_blocks(elements_of(payload))
    if not blocks:
        return "", {}
    text, _questions, fields = _render_block(blocks[0], keep_answers=True)
    return text, fields


def held_out_questions(payload: Any) -> list[dict[str, Any]]:
    """Wave-4 items. Selected answers are returned as gold, not inside the prompt."""
    questions: list[dict[str, Any]] = []
    for block in iter_blocks(elements_of(payload)):
        _text, rendered, _fields = _render_block(block, keep_answers=False)
        questions.extend(rendered)
    return questions


def _render_block(
    block: dict[str, Any],
    *,
    keep_answers: bool,
) -> tuple[str, list[dict[str, Any]], dict[str, str]]:
    lines: list[str] = []
    questions: list[dict[str, Any]] = []
    fields: dict[str, str] = {}
    name = strip_html(block.get("BlockName") or "Block")
    if name:
        lines.append(name)
    raw_questions = block.get("Questions") or []
    if not isinstance(raw_questions, list):
        return "\n".join(lines).strip(), questions, fields
    for index, question in enumerate(raw_questions, start=1):
        if not isinstance(question, dict):
            continue
        rendered, pieces = _render_question(question, index=index, keep_answers=keep_answers)
        if rendered:
            lines.append(rendered)
        questions.extend(pieces)
        if keep_answers:
            for piece in pieces:
                if piece.get("gold"):
                    fields[piece["id"]] = str(piece["gold"])
    return "\n".join(lines).strip(), questions, fields


def _render_question(
    question: dict[str, Any],
    *,
    index: int,
    keep_answers: bool,
) -> tuple[str, list[dict[str, Any]]]:
    prompt = strip_html(question.get("QuestionText") or "")
    qid = strip_html(question.get("QuestionID") or f"q{index}") or f"q{index}"
    kind = strip_html(question.get("QuestionType") or "text") or "text"
    if kind == "DB":
        return "", []
    options = option_labels(question)
    gold = answer_label(question)
    rows = question.get("Rows") if kind == "Matrix" else None
    if isinstance(rows, list) and rows:
        return _render_matrix(question, qid=qid, prompt=prompt, options=options, keep_answers=keep_answers)
    shown = _prompt_with_options(prompt, options)
    record = {
        "id": qid,
        "prompt": shown,
        "options": list(options),
        "kind": _kind(kind),
        "gold": gold,
    }
    if keep_answers and gold:
        shown = f"{shown}\nAnswer: {gold}" if shown else f"Answer: {gold}"
    return shown, [record]


def _render_matrix(
    question: dict[str, Any],
    *,
    qid: str,
    prompt: str,
    options: tuple[str, ...],
    keep_answers: bool,
) -> tuple[str, list[dict[str, Any]]]:
    rows = question.get("Rows") or []
    answers = question.get("Answers") if isinstance(question.get("Answers"), dict) else {}
    selected = answers.get("SelectedText") if isinstance(answers, dict) else None
    if not isinstance(selected, list):
        selected = []
    lines = [prompt] if prompt else []
    pieces: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        row_text = strip_html(row)
        if not row_text:
            continue
        gold = strip_html(selected[index]) if index < len(selected) else ""
        row_prompt = _prompt_with_options(f"{prompt}\n{row_text}".strip(), options)
        pieces.append(
            {
                "id": f"{qid}:{index + 1}",
                "prompt": row_prompt,
                "options": list(options),
                "kind": "likert" if options else "text",
                "gold": gold,
            }
        )
        line = row_text if not (keep_answers and gold) else f"{row_text}: {gold}"
        lines.append(line)
    return "\n".join(lines).strip(), pieces


def _prompt_with_options(prompt: str, options: tuple[str, ...]) -> str:
    if not options:
        return prompt
    listed = "\n".join(f"- {option}" for option in options)
    if prompt:
        return f"{prompt}\nOptions:\n{listed}"
    return f"Options:\n{listed}"


def _kind(question_type: str) -> str:
    folded = question_type.casefold()
    if folded in {"mc", "multiple choice"}:
        return "multiple_choice"
    if folded == "matrix":
        return "likert"
    if folded == "slider":
        return "numeric"
    return "text"
