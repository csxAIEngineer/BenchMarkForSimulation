"""Full-sample need dose and budget sentences for the two-option pricing stem.

Each arm inserts one fixed sentence into every question. The persona card
is not edited. Lengths are locked: N1 50, N2 49, N3 48, N4 51, BUD 52.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from popbench.evaluate.stimulus import parse_stimulus
from popbench.simulate.need_inject import TWO_OPTION_BLOCK, buy_code, parse_two_option

DOSE_SENTENCES = {
    "N1": "You bought it last week, so you still have plenty.",
    "N2": "You bought it a while ago, so you have some left.",
    "N3": "You will run out of it soon, so you have little.",
    "N4": "You ran out of it last week, so you have none left.",
    "BUD": "You ran out of it, and have only $20 left this week.",
}
DOSE_LENGTHS = {"N1": 50, "N2": 49, "N3": 48, "N4": 51, "BUD": 52}
DOSE_LEVEL = {"N1": 1, "N2": 2, "N3": 3, "N4": 4}
PRICE_CLAUSE = re.compile(r"The product is priced at: \$\d+\.\d{2}\.")
_PRICE_THEN_QUESTION = re.compile(
    r"(The product is priced at: \$\d+\.\d{2}\.) Would you or would you not purchase this product\?"
)


def assert_dose_lengths() -> None:
    """Stop if a locked sentence is not the required character count."""
    for arm, sentence in DOSE_SENTENCES.items():
        if len(sentence) != DOSE_LENGTHS[arm]:
            raise ValueError(f"{arm} length is {len(sentence)}, required {DOSE_LENGTHS[arm]}")
        if arm == "BUD" and "$20" not in sentence:
            raise ValueError("BUD sentence must keep $20")


def inject_dose(stem: str, need_arm: str) -> str:
    """Insert one locked sentence between the price and ``Would you``."""
    assert_dose_lengths()
    if need_arm not in DOSE_SENTENCES:
        raise ValueError(f"unknown need arm {need_arm!r}")
    match = _PRICE_THEN_QUESTION.search(stem)
    if match is None:
        raise ValueError("pricing stem has no price immediately before the purchase question")
    if stem[match.end(1)] != " ":
        raise ValueError("expected one space between the price and the question")
    sentence = DOSE_SENTENCES[need_arm]
    injected = stem[: match.end(1)] + " " + sentence + " " + stem[match.end(1) + 1 :]
    if injected.count(sentence) != 1:
        raise ValueError("need sentence must appear exactly once")
    if len(PRICE_CLAUSE.findall(injected)) != 1:
        raise ValueError("price clause must appear exactly once")
    if not injected.endswith("\n"):
        injected += "\n"
    return injected + "\n" + TWO_OPTION_BLOCK


def annotate_need_dose(
    rows: Sequence[Mapping[str, object]],
    items_by_question: Mapping[str, Mapping[str, object]],
    triplets: Mapping[tuple[str, str], Mapping[str, object]],
    *,
    need_arm: str,
    full_raw: str,
) -> list[dict[str, object]]:
    """Attach the locked triplet and the arm sentence. Unmapped rows stay."""
    sentence = DOSE_SENTENCES[need_arm]
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
                "need_arm": need_arm,
                "parse_error": parse_error,
                "parsed_choice": parsed,
                "pid": subject_id,
                "price": price,
                "product_id": product_id,
                "question_id": question_id,
                "raw_text": raw_text,
                "sentence": sentence,
                "subject_id": subject_id,
                "tier": int(locked["tier"]),
                "unmapped_reason": reason,
            }
        )
    return annotated
