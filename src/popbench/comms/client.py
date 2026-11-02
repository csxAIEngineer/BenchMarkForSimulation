"""Send a prompt package to a provider, with cache, retry, and JSON parsing."""

from __future__ import annotations

from pathlib import Path

from popbench.schema import PromptPackage, SimulationResult
from popbench.comms.cache import cache_key, read_cache, write_cache
from popbench.comms.parse import ParseError, parse_json_object
from popbench.comms.providers.base import Provider
from popbench.comms.retry import call_with_retry


class LLMClient:
    def __init__(
        self,
        provider: Provider,
        *,
        model: str,
        cache_dir: Path | None = None,
        attempts: int = 3,
        temperature: float | None = 0,
    ) -> None:
        self.provider = provider
        self.model = model
        self.cache_dir = cache_dir
        self.attempts = attempts
        self.temperature = temperature

    def complete(self, package: PromptPackage) -> SimulationResult:
        key = cache_key(
            package,
            provider=self.provider.name,
            model=self.model,
            temperature=self.temperature,
        )
        cached_raw = None if self.cache_dir is None else read_cache(self.cache_dir, key)
        if cached_raw is not None:
            parsed, error = _parse(cached_raw)
            return _result(package, self.provider.name, self.model, cached_raw, parsed, error, cached=True)
        raw = call_with_retry(
            lambda: self.provider.complete(
                package.messages(),
                model=self.model,
                temperature=self.temperature,
            ),
            attempts=self.attempts,
        )
        parsed, error = _parse(raw)
        if self.cache_dir is not None and error is None:
            write_cache(self.cache_dir, key, raw, parsed)
        return _result(package, self.provider.name, self.model, raw, parsed, error, cached=False)


def _parse(raw: str) -> tuple[dict | None, str | None]:
    try:
        return parse_json_object(raw), None
    except ParseError as exc:
        return None, str(exc)


def _result(
    package: PromptPackage,
    provider: str,
    model: str,
    raw: str,
    parsed: dict | None,
    error: str | None,
    *,
    cached: bool,
) -> SimulationResult:
    return SimulationResult(
        subject_id=package.subject_id,
        question_id=package.question_id,
        method=package.method,
        provider=provider,
        model=model,
        raw=raw,
        parsed=parsed,
        cached=cached,
        parse_error=error,
    )
