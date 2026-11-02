"""Retry a provider call. Parsing is not retried."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from popbench.comms.errors import CommsTransportError

T = TypeVar("T")


def call_with_retry(
    action: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.2,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    last: CommsTransportError | None = None
    for index in range(attempts):
        try:
            return action()
        except CommsTransportError as exc:
            last = exc
            if index + 1 == attempts:
                break
            sleep(base_delay * (2**index))
    assert last is not None
    raise last
