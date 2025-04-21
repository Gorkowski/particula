"""
Optional JIT compilation using Numba.
This module provides a decorator to optionally JIT-compile
functions using Numba.
"""
import functools
import os, sys
try:
    import numba

    HAVE_NUMBA = True
except ImportError:
    # If Numba is not installed, set to False by default
    HAVE_NUMBA = False


def numba_jit_wrapper(func):
    """Decorator to optionally JIT-compile the function `func`."""
    # Try to import Numba's JIT compiler

    compiled_func = None
    if HAVE_NUMBA:
        # Pre-compile the function in nopython mode for speed, if JIT is
        # enabled at some point
        compiled_func = numba.njit(
            func
        )  # compile now (can also do lazy compile later)

    # Define a wrapper that dispatches to compiled or original
    # based on the flag
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if HAVE_NUMBA and _jit_enabled() and compiled_func is not None:
            # Call the Numba-compiled version
            return compiled_func(*args, **kwargs)
        # Fallback to pure Python execution
        return func(*args, **kwargs)

    return wrapper
