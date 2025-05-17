"""
Smoke-tests for Taichi vapor–pressure strategies.

Checks that:
  • the private Taichi kernels (_xxx_kernel) execute without error, and
  • their results match the corresponding public python wrappers.
"""
import math
import numpy as np
import pytest

from particula.backend.taichi.gas.ti_vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
    AntoineVaporPressureStrategy,
    ClausiusClapeyronStrategy,
    WaterBuckStrategy,
)

_REL_TOL = 1e-10
_ABS_TOL = 1e-12
_T   = 300.0          # K
_M   = 0.018          # kg mol-1 (≈ water, arbitrary)
_CONC = 1.0           # mol m-3  (arbitrary)



def _assert_close(a, b):
    assert math.isclose(a, b, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)


@pytest.mark.parametrize(
    "strategy",
    [
        ConstantVaporPressureStrategy(1.0e3),                      # constant VP
        AntoineVaporPressureStrategy(1.0, 1.0, 1.0),               # dummy coefficients
        ClausiusClapeyronStrategy(4.0e4, _T, 1.0e5),               # latent-heat etc.
        WaterBuckStrategy(),                                       # Buck water
    ],
)
def test_kernels_vs_wrappers(strategy):
    """Ensure kernels & wrappers give identical numerical outputs."""
    # ── pure vapor pressure ───────────────────────────────────────────────
    vp_kernel  = strategy._pure_vp_kernel(_T)
    vp_wrapper = strategy.pure_vapor_pressure(_T)
    _assert_close(vp_kernel, vp_wrapper)

    # ── partial pressure ─────────────────────────────────────────────────
    pp_kernel  = strategy._partial_pressure_kernel(_CONC, _M, _T)
    pp_wrapper = strategy.partial_pressure(_CONC, _M, _T)
    _assert_close(pp_kernel, pp_wrapper)

    # ── concentration from pressure ──────────────────────────────────────
    c_kernel   = strategy._concentration_kernel(pp_wrapper, _M, _T)
    c_wrapper  = strategy.concentration(pp_wrapper, _M, _T)
    _assert_close(c_kernel, c_wrapper)
