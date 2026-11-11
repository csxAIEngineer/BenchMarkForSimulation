"""Persona card plus pricing questions. Human answers stay out of the prompt."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

WTP_PRODUCTS: dict[str, str] = {
    "local_field": "local field-grown lettuce",
    "local_greenhouse": "local greenhouse-grown lettuce",
    "local_vertical": "local vertically farmed lettuce",
    "nonlocal_field": "non-local field-grown lettuce",
    "nonlocal_greenhouse": "non-local greenhouse-grown lettuce",
    "nonlocal_vertical": "non-local vertically farmed lettuce",
}

SYSTEM = (
    "You answer as the person in the persona card. The card is copied from a survey. "
    "Do not add a life history. Reply with one JSON object and no other text:\n"
    '{"answers":[{"question_id":"...","answer":"..."}]}\n'
    "Copy each question_id exactly. Put only the answer in answer."
)


def question_prompt(item: Mapping[str, object]) -> str:
    """Stimulus text for one held-out item. The human answer is not included."""
    question_id = str(item.get("question_id") or "")
    product = WTP_PRODUCTS.get(question_id)
    if product is not None:
        return (
            "You are taking part in a Becker-DeGroot-Marschak elicitation in Ontario, Canada. "
            f"What is the most you would pay, in Canadian dollars from 0 to 10, for {product}?"
        )
    if question_id == "conclusion_purchased_product":
        stimulus = item.get("stimulus") if isinstance(item.get("stimulus"), Mapping) else {}
        selected = str(stimulus.get("conclusion_selected_product") or "").strip() or "the randomly selected lettuce"
        price = str(stimulus.get("conclusion_generated_price") or "").strip() or "the randomly drawn price"
        return (
            "Under the same Becker-DeGroot-Marschak rule, one product was drawn at random "
            f"({selected}) and a price was drawn at random ({price} Canadian dollars). "
            "Would you purchase that product at the drawn price? Answer TRUE or FALSE."
        )
    text = item.get("question_text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{question_id or 'item'} has no question text")
    return text


def build_messages(
    card_text: str,
    questions: Sequence[tuple[str, str]],
    *,
    system: str | None = None,
) -> list[dict[str, str]]:
    """One chat request: the card, then each question id and its stimulus."""
    if not card_text.strip():
        raise ValueError("persona card text is empty")
    if not questions:
        raise ValueError("a simulation request needs at least one question")
    lines = ["## Persona", card_text.rstrip(), "", "## Questions"]
    seen: set[str] = set()
    for question_id, prompt in questions:
        if not question_id or question_id in seen:
            raise ValueError(f"duplicate or empty question_id: {question_id!r}")
        seen.add(question_id)
        if not prompt.strip():
            raise ValueError(f"{question_id} prompt is empty")
        lines.extend((f"question_id: {question_id}", prompt.rstrip(), ""))
    user = "\n".join(lines).rstrip() + "\n"
    return [
        {"role": "system", "content": SYSTEM if system is None else system},
        {"role": "user", "content": user},
    ]
