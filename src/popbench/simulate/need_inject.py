"""Need-state sentence inserted into the locked two-option pricing stem.

The persona card is not edited. The only change to the wave-4 sentence is
one space, the need sentence, and one space between the price and
``Would you``.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence

from popbench.evaluate.options import NO_OPTION, YES_OPTION, match_option
from popbench.evaluate.stimulus import parse_stimulus

NEED_SENTENCES = {
    "high": "You ran out of it last week, so you have none left.",
    "low": "You bought it last week, so you still have plenty.",
    "fill": "This product is kept on a shelf by the store exit.",
}
NEED_ORDER = ("high", "low", "fill", "fill")

TWO_OPTION_BLOCK = (
    "Options:\n"
    "A) Yes, I would purchase the product\n"
    "B) No, I would not purchase the product\n"
)

_PRICE_THEN_QUESTION = re.compile(
    r"(The product is priced at: \$\d+\.\d{2}\.) Would you or would you not purchase this product\?"
)
_LETTER = re.compile(r"[\s\(\*]*([ABab])[\s\)\.\*]*")
_NAMED_LETTER = re.compile(r"(?:option|answer|choice)\s*[:\-]?\s*\(?([ABab])\)?\b", re.IGNORECASE)
_LEADING_LETTER = re.compile(r"^\(?([ABab])\)?[\s\.\):]")


def sentence_lengths_ok() -> bool:
    """Every locked sentence is 50 or 51 characters."""
    return all(50 <= len(text) <= 51 for text in NEED_SENTENCES.values())


def need_state(pid: str, product_id: str) -> str:
    """Deterministic high, low, or fill. Fill is twice as likely as either extreme."""
    digest = hashlib.sha256(f"{pid}|{product_id}|need".encode()).digest()
    return NEED_ORDER[digest[0] % 4]


def inject_need(stem: str, state: str) -> str:
    """Insert the need sentence inline, then append the two official options."""
    if state not in NEED_SENTENCES:
        raise ValueError(f"unknown need state {state!r}")
    if not sentence_lengths_ok():
        raise ValueError("need sentences must be 50 or 51 characters")
    match = _PRICE_THEN_QUESTION.search(stem)
    if match is None:
        raise ValueError("pricing stem has no price immediately before the purchase question")
    sentence = NEED_SENTENCES[state]
    injected = stem[: match.end(1)] + " " + sentence + " " + stem[match.end(1) + 1 :]
    if not injected.endswith("\n"):
        injected += "\n"
    return injected + "\n" + TWO_OPTION_BLOCK


def parse_two_option(text: str | None) -> str | None:
    """Map a reply to A or B with the official option matcher, then a lone letter.

    Words such as need or expensive are not evidence. A or B is the label on
    the two-option block, not a third semantic class.
    """
    if not text:
        return None
    matched = match_option(text, [YES_OPTION, NO_OPTION])
    if matched == YES_OPTION:
        return "A"
    if matched == NO_OPTION:
        return "B"
    stripped = text.strip()
    full = _LETTER.fullmatch(stripped)
    if full:
        return full.group(1).upper()
    named = _NAMED_LETTER.search(stripped)
    if named:
        return named.group(1).upper()
    leading = _LEADING_LETTER.match(stripped)
    if leading:
        return leading.group(1).upper()
    return None


def buy_code(parsed_choice: str | None) -> int | None:
    if parsed_choice == "A":
        return 1
    if parsed_choice == "B":
        return 0
    return None


def annotate_need_inject(
    rows: Sequence[Mapping[str, object]],
    items_by_question: Mapping[str, Mapping[str, object]],
    triplets: Mapping[tuple[str, str], Mapping[str, object]],
    *,
    full_raw: str,
) -> list[dict[str, object]]:
    """Attach the locked triplet and the need state. Unmapped rows stay."""
    annotated: list[dict[str, object]] = []
    for row in rows:
        question_id = str(row["question_id"])
        subject_id = str(row["subject_id"])
        locked = triplets.get((subject_id, question_id))
        if locked is None:
            raise ValueError(f"{subject_id} {question_id} is not in the locked triplets")
        item = items_by_question[question_id]
        stimulus = parse_stimulus(str(item.get("question_text") or ""))
        if stimulus is None:
            raise ValueError(f"{question_id} is not a locked pricing stimulus")
        product_id = str(locked["product_id"])
        price = str(locked["price"])
        if str(stimulus["product"]) != product_id or f"{stimulus['price_usd']:.2f}" != price:
            raise ValueError(f"{subject_id} {question_id} does not match the locked triplet")
        state = need_state(subject_id, product_id)
        answer_text = "" if row.get("answer") is None else str(row.get("answer"))
        parse_error = row.get("parse_error")
        if parse_error:
            parsed = None
            reason: str | None = str(parse_error)
            raw_text = answer_text or full_raw
        else:
            parsed = parse_two_option(answer_text)
            reason = None if parsed is not None else "not a unique match to one held-out option"
            raw_text = answer_text
        annotated.append(
            {
                "answer": answer_text,
                "buy": buy_code(parsed),
                "dataset": row["dataset"],
                "model": row["model"],
                "need_state": state,
                "parse_error": parse_error,
                "parsed_choice": parsed,
                "pid": subject_id,
                "price": price,
                "product_id": product_id,
                "question_id": question_id,
                "raw_text": raw_text,
                "subject_id": subject_id,
                "tier": int(locked["tier"]),
                "unmapped_reason": reason,
            }
        )
    return annotated
