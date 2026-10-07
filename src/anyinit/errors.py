"""Exception hierarchy for AnyInit.

Every failure AnyInit raises derives from :class:`AnyInitError`, so callers can
guard a whole ``initialize`` call with one ``except``.
"""

from __future__ import annotations


class AnyInitError(Exception):
    """Base class for every AnyInit failure."""


class BackendNotFoundError(AnyInitError):
    """No registered backend recognized the object passed as a model."""


class BackendUnavailableError(AnyInitError):
    """A backend recognized the model but its framework could not be imported."""


class TraceError(AnyInitError):
    """The model's graph could not be captured at full fidelity."""


class ConfigError(AnyInitError, ValueError):
    """An argument to ``initialize`` is invalid.  Raised before the model is touched."""
