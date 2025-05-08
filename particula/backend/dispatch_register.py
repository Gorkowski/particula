"""
Backend dispatching and registration.
This module provides a mechanism for dispatching function calls to
accelerated implementations based on the currently-selected backend.
"""

from __future__ import annotations

import functools
from collections import defaultdict
from contextlib import contextmanager
from typing import Callable, Dict

# ----------------------------------------------------------------------
# _GLOBAL STATE
# ----------------------------------------------------------------------
_backend: str = "python"  # active backend
_registry: Dict[str, Dict[str, Callable]] = defaultdict(
    dict
)  # name → {backend: impl}

# ----------------------------------------------------------------------
# PUBLIC HELPERS
# ----------------------------------------------------------------------


def use_backend(name: str = "python") -> None:
    """
    Set the active backend.  Pass ``"python"`` (or leave empty) to disable
    acceleration globally.
    """
    global _backend
    _backend = (name or "python").lower()


def get_backend() -> str:
    """Return the currently-selected backend name."""
    return _backend


@contextmanager
def backend(name: str):
    """
    Temporary backend switch:

        with backend("taichi"):
            # runs under Taichi
            ...
    """
    prev = _backend
    use_backend(name)
    try:
        yield
    finally:
        use_backend(prev)


# ----------------------------------------------------------------------
# DECORATORS
# ----------------------------------------------------------------------
def dispatchable(func: Callable) -> Callable:
    """
    Decorator for the *reference* Python implementation.

    The returned wrapper looks up an accelerated implementation for the
    currently-active backend; if none exists it falls back to *func*.
    The original Python implementation is always registered under the key
    ``"python"``.
    """
    func_name = func.__name__
    _registry.setdefault(func_name, {})["python"] = func  # default path

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        impl = _registry.get(func_name, {}).get(_backend, func)
        return impl(*args, **kwargs)

    return wrapper


def register(func_name: str, *, backend: str) -> Callable:
    """
    Decorator factory for accelerated implementations.

    Usage
    -----
        @register("coagulation_gain_rate", backend="taichi")
        def gain_rate_taichi(...):
            ...

    The implementation is stored in the global registry and automatically
    discovered by @dispatchable wrappers with the same *func_name*.
    """
    backend = backend.lower()

    def decorator(accel_func: Callable) -> Callable:
        _registry.setdefault(func_name, {})[backend] = accel_func
        return accel_func

    return decorator


# ----------------------------------------------------------------------
# WHAT WE EXPORT
# ----------------------------------------------------------------------
__all__ = [
    "use_backend",
    "get_backend",
    "backend",
    "dispatchable",
    "register",
]
