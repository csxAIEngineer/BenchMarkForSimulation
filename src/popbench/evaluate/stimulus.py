"""Parse the Twin2K wave-4 pricing stimulus.

The wave JSON and the label CSV do not have a ``price_shown`` column.
``QID9_n`` is the nth offer shown to that person, not a fixed product.
The copied ``QuestionText`` is the stimulus. In the locked sample every
item uses one sentence, and the currency amount is the ``$X.XX`` slot:

``The product is priced at: $X.XX.``

Amounts are US dollars with two decimal places. ``$0.00`` is the 0% step
of that product's own shelf-price grid, not a blank and not a missing-value
code. A text that does not match the sentence is unparsed and is never
treated as free.

For product ``p``, let ``M_p`` be the largest positive parsed price of ``p``
in the offers under analysis. Tier ``k`` for ``k = 0..10`` is
``round_half_up(k/10 * M_p)`` to the cent. Tier 0 is free. Tiers are
fractions of that product's own maximum, so dollar prices are not pooled
across products.
"""

from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP

_STIMULUS = re.compile(
    r"^Please consider the following product category: (?P<category>.+?)\. "
    r"Suppose you are in a grocery store, and you see the following product in that category: "
    r"(?P<product>.+?)\. The product is priced at: \$(?P<amount>\d+\.\d{2})\. "
    r"Would you or would you not purchase this product\?$"
)


def parse_stimulus(question_text: str) -> dict[str, object] | None:
    """Return category, product, and USD price, or None if the sentence does not match."""
    match = _STIMULUS.match(question_text.strip())
    if match is None:
        return None
    amount = Decimal(match.group("amount"))
    if amount < 0:
        return None
    return {
        "category": match.group("category"),
        "price_usd": amount,
        "product": match.group("product"),
    }


def is_free_price(price_usd: Decimal) -> bool:
    """True only for an exact zero currency amount."""
    return price_usd == Decimal("0.00")


def cents(price_usd: Decimal) -> Decimal:
    return price_usd.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def tier_index(price_usd: Decimal, product_max_usd: Decimal) -> int | None:
    """Map a price onto k/10 of the product maximum. None if it is off the grid."""
    if product_max_usd <= 0:
        return None
    for step in range(0, 11):
        expected = cents(product_max_usd * Decimal(step) / Decimal(10))
        if price_usd == expected:
            return step
    return None
