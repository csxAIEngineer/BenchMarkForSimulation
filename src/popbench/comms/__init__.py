"""Communication layer: providers, retry, cache, and JSON parsing.

Persona methods never import this package. Callers pass an already built
``PromptPackage``.
"""

from popbench.comms.client import LLMClient
from popbench.comms.providers.cr import CRProvider
from popbench.comms.providers.mock import MockProvider
from popbench.comms.providers.openai_compat import OpenAICompatProvider

PROVIDERS = {
    MockProvider.name: MockProvider,
    "openai": OpenAICompatProvider,
    "cr": CRProvider,
}


def build_provider(name: str, **kwargs: object):
    if name not in PROVIDERS:
        known = ", ".join(sorted(PROVIDERS))
        raise KeyError(f"unknown provider {name!r}. Known providers: {known}")
    return PROVIDERS[name](**kwargs)


__all__ = [
    "LLMClient",
    "PROVIDERS",
    "build_provider",
]
