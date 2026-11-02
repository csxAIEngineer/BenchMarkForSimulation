"""Communication-layer errors. Persona methods do not raise these."""


class CommsError(RuntimeError):
    """The communication layer could not complete a request."""


class CommsTransportError(CommsError):
    """The provider call failed and may be retried."""
