"""Send a prompt package to a provider, with cache, retry, and JSON parsing."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from popbench.schema import PromptPackage, SimulationResult
from popbench.comms.batch import messages_for_batch, pack_batch, per_question_payloads
from popbench.comms.cache import cache_key, cache_key_messages, read_cache, write_cache
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

    def complete_batch(self, packages: Sequence[PromptPackage]) -> tuple[SimulationResult, ...]:
        """One provider call for several questions that share a persona."""
        if not packages:
            return ()
        messages = messages_for_batch(packages)
        _system, _user, question_ids = pack_batch(packages)
        mode = packages[0].response_mode
        key = cache_key_messages(
            messages,
            provider=self.provider.name,
            model=self.model,
            temperature=self.temperature,
            response_mode=mode,
        )
        cached_raw = None if self.cache_dir is None else read_cache(self.cache_dir, key)
        if cached_raw is not None:
            return self._rows(packages, question_ids, cached_raw, cached=True)
        raw = call_with_retry(
            lambda: self.provider.complete(
                messages,
                model=self.model,
                temperature=self.temperature,
            ),
            attempts=self.attempts,
        )
        rows = self._rows(packages, question_ids, raw, cached=False)
        if self.cache_dir is not None and all(row.parse_error is None for row in rows):
            parsed, _error = _parse(raw)
            write_cache(self.cache_dir, key, raw, parsed)
        return rows

    def _rows(
        self,
        packages: Sequence[PromptPackage],
        question_ids: tuple[str, ...],
        raw: str,
        *,
        cached: bool,
    ) -> tuple[SimulationResult, ...]:
        parsed, error = _parse(raw)
        payloads = per_question_payloads(
            parsed,
            question_ids,
            response_mode=packages[0].response_mode,
            parse_error=error,
        )
        rows: list[SimulationResult] = []
        for package, (item, item_error) in zip(packages, payloads, strict=True):
            rows.append(
                SimulationResult(
                    subject_id=package.subject_id,
                    question_id=package.question_id,
                    method=package.method,
                    provider=self.provider.name,
                    model=self.model,
                    raw=raw,
                    parsed=item,
                    cached=cached,
                    parse_error=item_error,
                )
            )
        return tuple(rows)


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
