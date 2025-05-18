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
_TEMPERATURE      = 300.0         # K
_MOLAR_MASS      = 0.018         # kg / mol   (arbitrary)
_CONCENTRATION   = 4.0           # mol / m3   (arbitrary)


def _close(a, b):
    assert math.isclose(a, b, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)


def _check_kernels_vs_wrappers(strategy):
    """Helper: make sure kernels == python wrappers for ONE Taichi strategy."""
    _close(strategy._pure_vp_kernel(_TEMPERATURE),                            # pure-VP
           strategy.pure_vapor_pressure(_TEMPERATURE))
    _close(strategy._partial_pressure_kernel(_CONCENTRATION, _MOLAR_MASS, _TEMPERATURE),        # P-part
           strategy.partial_pressure(_CONCENTRATION, _MOLAR_MASS, _TEMPERATURE))
    _close(strategy._concentration_kernel(                          # conc.
               strategy.pure_vapor_pressure(_TEMPERATURE), _MOLAR_MASS, _TEMPERATURE),
           strategy.concentration(strategy.pure_vapor_pressure(_TEMPERATURE), _MOLAR_MASS, _TEMPERATURE))


def _check_taichi_vs_python(ti_strategy, py_strategy):
    """Helper: compare numerical results Taichi ↔ python strategy."""
    _close(ti_strategy.pure_vapor_pressure(_TEMPERATURE),
           py_strategy.pure_vapor_pressure(_TEMPERATURE))
    _close(ti_strategy.partial_pressure(_CONCENTRATION, _MOLAR_MASS, _TEMPERATURE),
           py_strategy.partial_pressure(_CONCENTRATION, _MOLAR_MASS, _TEMPERATURE))
    _close(ti_strategy.concentration(
               ti_strategy.pure_vapor_pressure(_TEMPERATURE), _MOLAR_MASS, _TEMPERATURE),
           py_strategy.concentration(
               py_strategy.pure_vapor_pressure(_TEMPERATURE), _MOLAR_MASS, _TEMPERATURE))

# ── individual strategy tests ──────────────────────────────────────────────
def test_constant_strategy():
    const_p = 1.0e3
    ti_strategy = TiConstant(const_p)
    py_strategy = PyConstant(const_p)
    _check_kernels_vs_wrappers(ti_strategy)
    _check_taichi_vs_python(ti_strategy, py_strategy)


def test_antoine_strategy():
    coeff_a, coeff_b, coeff_c = 1.0, 2.0, 9.0     # arbitrary coefficients
    ti_strategy = TiAntoine(coeff_a, coeff_b, coeff_c)
    py_strategy = PyAntoine(coeff_a, coeff_b, coeff_c)
    _check_kernels_vs_wrappers(ti_strategy)
    _check_taichi_vs_python(ti_strategy, py_strategy)


def test_clausius_clapeyron_strategy():
    latent_heat = 4.0e4
    temperature_reference    = _TEMPERATURE
    pressure_reference   = 1.0e5
    ti_strategy = TiCC(latent_heat, temperature_reference, pressure_reference)
    py_strategy = PyCC(latent_heat, temperature_reference, pressure_reference)
    _check_kernels_vs_wrappers(ti_strategy)
    _check_taichi_vs_python(ti_strategy, py_strategy)


def test_buck_water_strategy():
    ti_strategy = TiBuck()
    py_strategy = PyBuck()
    _check_kernels_vs_wrappers(ti_strategy)
    _check_taichi_vs_python(ti_strategy, py_strategy)
