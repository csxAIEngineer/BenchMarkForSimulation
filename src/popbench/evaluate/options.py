"""Deterministic purchase-option matching for Twin2K pricing items.

Held-out answers are full option sentences, for example
``Yes, I would purchase the product`` and
``No, I would not purchase the product``. Model replies are often a short
form (``Yes``, ``No``, ``Would purchase``). Exact string match therefore
disagrees even when the reply names the same option.

A reply is mapped onto one held-out option only when the match is unique:

- the reply tokens equal that option, or
- they are a prefix of exactly one option, or
- the leading token is ``yes`` or ``no`` and exactly one option starts
  with that token, or
- the whole reply is a contiguous word sequence from exactly one option.

The single substitution of ``this`` to ``the`` is tried only for that same
prefix or contiguous-sequence test. Anything else, including bare ``Would``
(a word shared by both options) and answers that add commentary, stays
unmapped. Polarity is then read from the matched option, not re-inferred
from leftover words: a leading ``yes`` selects the Yes option and codes 1,
a leading ``no`` selects the No option and codes 0.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")

YES_OPTION = "Yes, I would purchase the product"
NO_OPTION = "No, I would not purchase the product"


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


def _contiguous(needle: list[str], hay: list[str]) -> bool:
    if not needle or len(needle) > len(hay):
        return False
    width = len(needle)
    return any(hay[index : index + width] == needle for index in range(len(hay) - width + 1))


def _prefix(needle: list[str], hay: list[str]) -> bool:
    return bool(needle) and len(needle) <= len(hay) and hay[: len(needle)] == needle


def match_option(answer: str, options: list[str]) -> str | None:
    """Return the held-out option text, or None when the reply is not a unique match."""
    answer_tokens = tokens(answer)
    if not answer_tokens:
        return None
    parsed = [(option, tokens(option)) for option in options if tokens(option)]
    if not parsed:
        return None

    def unique(hits: list[str]) -> str | None:
        if len(hits) == 1:
            return hits[0]
        return None

    exact = unique([option for option, option_tokens in parsed if option_tokens == answer_tokens])
    if exact:
        return exact

    prefixed = [option for option, option_tokens in parsed if _prefix(answer_tokens, option_tokens)]
    if len(prefixed) == 1:
        return prefixed[0]
    if len(prefixed) > 1:
        return None

    if answer_tokens[0] in {"yes", "no"}:
        return unique([option for option, option_tokens in parsed if option_tokens[0] == answer_tokens[0]])

    sliced = [option for option, option_tokens in parsed if _contiguous(answer_tokens, option_tokens)]
    if len(sliced) == 1:
        return sliced[0]
    if len(sliced) > 1:
        return None

    if "this" in answer_tokens:
        swapped = ["the" if token == "this" else token for token in answer_tokens]
        swapped_hits = [
            option
            for option, option_tokens in parsed
            if _contiguous(swapped, option_tokens) or _prefix(swapped, option_tokens)
        ]
        return unique(swapped_hits)
    return None


def option_polarity(option: str) -> int | None:
    """Code one canonical option: Yes-like purchase = 1, No-like refusal = 0."""
    option_tokens = tokens(option)
    if not option_tokens:
        return None
    if option_tokens[0] == "yes":
        return 1
    if option_tokens[0] == "no":
        return 0
    if "not" in option_tokens:
        return 0
    if "purchase" in option_tokens:
        return 1
    return None


def binary_codebook(options: list[str]) -> dict[str, int] | None:
    """Map each option string to 0 or 1. None unless both polarities are present."""
    coded: dict[str, int] = {}
    for option in options:
        polarity = option_polarity(option)
        if polarity is None:
            return None
        coded[option] = polarity
    if set(coded.values()) != {0, 1}:
        return None
    return coded


def code_purchase(answer: str, codebook: dict[str, int]) -> int | None:
    """Map a reply onto 1 or 0 through ``codebook``. None if it does not match."""
    matched = match_option(answer, list(codebook))
    if matched is None:
        return None
    return codebook[matched]
