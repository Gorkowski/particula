"""
Smoke tests for Taichi vapor–pressure strategies.

Tests performed for every strategy
  • low-level kernels vs. public python wrappers (Taichi side)
  • Taichi numerical result vs. reference pure-python implementation
"""

import math
import pytest

# Taichi strategies
from particula.backend.taichi.gas.ti_vapor_pressure_strategies import (
    ConstantVaporPressureStrategy as TiConstant,
    AntoineVaporPressureStrategy  as TiAntoine,
    ClausiusClapeyronStrategy     as TiCC,
    WaterBuckStrategy             as TiBuck,
)

# Reference python strategies
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy as PyConstant,
    AntoineVaporPressureStrategy  as PyAntoine,
    ClausiusClapeyronStrategy     as PyCC,
    WaterBuckStrategy             as PyBuck,
)

# ────────────────────────────────────────────────────────────────────────────
_REL_TOL = 1e-9
_ABS_TOL = 1e-12
_T      = 300.0         # K
_M      = 0.018         # kg / mol   (arbitrary)
_CONC   = 4.0           # mol / m3   (arbitrary)


def _close(a, b):
    assert math.isclose(a, b, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)


def _check_kernels_vs_wrappers(strategy):
    """Helper: make sure kernels == python wrappers for ONE Taichi strategy."""
    _close(strategy._pure_vp_kernel(_T),                            # pure-VP
           strategy.pure_vapor_pressure(_T))
    _close(strategy._partial_pressure_kernel(_CONC, _M, _T),        # P-part
           strategy.partial_pressure(_CONC, _M, _T))
    _close(strategy._concentration_kernel(                          # conc.
               strategy.pure_vapor_pressure(_T), _M, _T),
           strategy.concentration(strategy.pure_vapor_pressure(_T), _M, _T))


def _check_taichi_vs_python(ti_strategy, py_strategy):
    """Helper: compare numerical results Taichi ↔ python strategy."""
    _close(ti_strategy.pure_vapor_pressure(_T),
           py_strategy.pure_vapor_pressure(_T))
    _close(ti_strategy.partial_pressure(_CONC, _M, _T),
           py_strategy.partial_pressure(_CONC, _M, _T))
    _close(ti_strategy.concentration(
               ti_strategy.pure_vapor_pressure(_T), _M, _T),
           py_strategy.concentration(
               py_strategy.pure_vapor_pressure(_T), _M, _T))

# ── individual strategy tests ──────────────────────────────────────────────
def test_constant_strategy():
    const_p = 1.0e3
    ti_s = TiConstant(const_p)
    py_s = PyConstant(const_p)
    _check_kernels_vs_wrappers(ti_s)
    _check_taichi_vs_python(ti_s, py_s)


def test_antoine_strategy():
    a, b, c = 1.0, 2.0, 9.0     # arbitrary coefficients
    ti_s = TiAntoine(a, b, c)
    py_s = PyAntoine(a, b, c)
    _check_kernels_vs_wrappers(ti_s)
    _check_taichi_vs_python(ti_s, py_s)


def test_clausius_clapeyron_strategy():
    latent_heat = 4.0e4
    temp_ref    = _T
    press_ref   = 1.0e5
    ti_s = TiCC(latent_heat, temp_ref, press_ref)
    py_s = PyCC(latent_heat, temp_ref, press_ref)
    _check_kernels_vs_wrappers(ti_s)
    _check_taichi_vs_python(ti_s, py_s)


def test_buck_water_strategy():
    ti_s = TiBuck()
    py_s = PyBuck()
    _check_kernels_vs_wrappers(ti_s)
    _check_taichi_vs_python(ti_s, py_s)
