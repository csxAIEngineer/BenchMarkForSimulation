"""Errors raised while building a persona prompt. No network calls."""


class PersonaMethodError(ValueError):
    """This persona method cannot be applied to the given case."""
