"""Three-option refusal prompt, letter parser, and semantic restore.

The pricing stem stays the wave-4 sentence. Only the answer lines change.
Option order is a deterministic hash of person and product, so a letter is
not a semantic label until it is mapped back through ``option_order``.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal

from popbench.evaluate.options import NO_OPTION, YES_OPTION
from popbench.evaluate.stimulus import parse_stimulus, tier_index

BUY_TEXT = "Yes, I would purchase the product"
NO_NEED_TEXT = "No, I do not need this product"
TOO_EXPENSIVE_TEXT = "No, it is too expensive for me"

# Semantic codes are the permutation alphabet. They are not the displayed letters.
OPTION_TEXT = {
    "A": BUY_TEXT,
    "B": NO_NEED_TEXT,
    "C": TOO_EXPENSIVE_TEXT,
}
SEMANTIC = {
    "A": "buy",
    "B": "no_need",
    "C": "too_expensive",
}
PERMS: tuple[tuple[str, str, str], ...] = (
    ("A", "B", "C"),
    ("A", "C", "B"),
    ("B", "A", "C"),
    ("B", "C", "A"),
    ("C", "A", "B"),
    ("C", "B", "A"),
)

THREE_OPTION_SYSTEM = (
    "You answer as the person in the persona card. The card is copied from a survey. "
    "Do not add a life history. Reply with one JSON object and no other text:\n"
    '{"answers":[{"question_id":"...","answer":"..."}]}\n'
    "Copy each question_id exactly. The answer for each question is one letter, A, B, or C. "
    "Do not explain and do not give a reason."
)

_FULL_LETTER = re.compile(r"[\s\(\*]*([ABCabc])[\s\)\.\*]*")
_NAMED_LETTER = re.compile(r"(?:option|answer|choice)\s*[:\-]?\s*\(?([ABCabc])\)?\b", re.IGNORECASE)
_LEADING_LETTER = re.compile(r"^\(?([ABCabc])\)?[\s\.\):]")


def perm_for(pid: str, product_id: str) -> tuple[str, str, str]:
    """One of the six semantic orders. The same pair always returns the same order."""
    digest = hashlib.sha256(f"{pid}|{product_id}".encode()).digest()
    return PERMS[digest[0] % 6]


def option_lines(option_order: Sequence[str]) -> str:
    """Display letters A, B, and C in that order. ``option_order`` is semantic codes."""
    if tuple(option_order) not in PERMS:
        raise ValueError(f"option_order {option_order!r} is not one of the six permutations")
    return "\n".join(f"{letter}) {OPTION_TEXT[code]}" for letter, code in zip("ABC", option_order))


def three_option_prompt(stem: str, option_order: Sequence[str]) -> str:
    """Keep the wave-4 stem and append the three option lines. No extra instructions."""
    if not stem.strip():
        raise ValueError("pricing stem is empty")
    body = stem if stem.endswith("\n") else stem + "\n"
    return body + "\nOptions:\n" + option_lines(option_order) + "\n"


def parse_choice(text: str | None) -> str | None:
    """Return A, B, or C using only the letter patterns. Words such as need are not evidence."""
    if not text:
        return None
    stripped = text.strip()
    full = _FULL_LETTER.fullmatch(stripped)
    if full:
        return full.group(1).upper()
    named = _NAMED_LETTER.search(stripped)
    if named:
        return named.group(1).upper()
    leading = _LEADING_LETTER.match(stripped)
    if leading:
        return leading.group(1).upper()
    return None


def to_semantic(letter: str, option_order: Sequence[str]) -> str:
    """Map a displayed letter through ``option_order`` to buy, no_need, or too_expensive.

    ``option_order`` ("C", "A", "B") means display position A holds semantic C.
    A reply of A is therefore too_expensive.
    """
    position = "ABC".index(letter.upper())
    return SEMANTIC[option_order[position]]


def collapsed_binary(semantic: str | None, raw_answer: str) -> str:
    """Map buy to the held-out Yes sentence and both refusals to the No sentence.

    An unparsed reply stays as the raw text so official matching does not invent a code.
    """
    if semantic == "buy":
        return YES_OPTION
    if semantic in {"no_need", "too_expensive"}:
        return NO_OPTION
    return raw_answer


def annotate_three_option(
    rows: Sequence[Mapping[str, object]],
    items_by_question: Mapping[str, Mapping[str, object]],
    product_max: Mapping[str, Decimal],
    *,
    full_raw: str,
) -> list[dict[str, object]]:
    """Add triplet fields and restore semantics. Rows are not dropped."""
    annotated: list[dict[str, object]] = []
    for row in rows:
        question_id = str(row["question_id"])
        item = items_by_question[question_id]
        stimulus = parse_stimulus(str(item.get("question_text") or ""))
        if stimulus is None:
            raise ValueError(f"{question_id} is not a locked pricing stimulus")
        product_id = str(stimulus["product"])
        price = stimulus["price_usd"]
        if not isinstance(price, Decimal):
            raise ValueError(f"{question_id} price is not a decimal")
        maximum = product_max.get(product_id)
        if maximum is None:
            raise ValueError(f"{product_id} has no positive price in the locked sample")
        tier = tier_index(price, maximum)
        if tier is None:
            raise ValueError(f"{product_id} price {price} is off the product grid")
        option_order = perm_for(str(row["subject_id"]), product_id)
        answer_text = "" if row.get("answer") is None else str(row.get("answer"))
        parse_error = row.get("parse_error")
        if parse_error:
            letter = None
            semantic = None
            reason: str | None = str(parse_error)
            raw_text = answer_text or full_raw
        else:
            letter = parse_choice(answer_text)
            if letter is None:
                semantic = None
                reason = "no A/B/C letter match"
                raw_text = answer_text
            else:
                semantic = to_semantic(letter, option_order)
                reason = None
                raw_text = answer_text
        annotated.append(
            {
                "answer": collapsed_binary(semantic, answer_text),
                "dataset": row["dataset"],
                "model": row["model"],
                "option_order": list(option_order),
                "parse_error": parse_error,
                "parsed_letter": letter,
                "pid": str(row["subject_id"]),
                "price": f"{price:.2f}",
                "product_id": product_id,
                "question_id": question_id,
                "raw_text": raw_text,
                "semantic": semantic,
                "subject_id": str(row["subject_id"]),
                "tier": tier,
                "unmapped_reason": reason,
            }
        )
    return annotated
