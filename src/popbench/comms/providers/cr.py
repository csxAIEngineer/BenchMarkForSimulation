"""CR API chat provider. Wraps the repo ``cr_api`` client."""

from __future__ import annotations

import os

from popbench.comms.errors import CommsError, CommsTransportError


class CRProvider:
    name = "cr"

    def __init__(self, *, api_key: str | None = None, timeout: float = 300) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("CR_API_KEY", "").strip()
        if not self.api_key:
            raise CommsError("missing CR_API_KEY")
        self.timeout = timeout

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float | None = None,
    ) -> str:
        from cr_api.client import CRClient, CRError

        try:
            result = CRClient(api_key=self.api_key).chat(
                messages,
                model=model,
                temperature=temperature,
                timeout=self.timeout,
            )
        except CRError as exc:
            raise CommsTransportError(str(exc)) from exc
        return str(result.content or "")
