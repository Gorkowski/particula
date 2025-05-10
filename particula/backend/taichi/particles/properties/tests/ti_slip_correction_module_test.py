"""
Unit tests for Taichi Cunningham slip-correction.
"""
import numpy as np
import numpy.testing as npt
import taichi as ti

from particula.particles.properties.slip_correction_module import (
    get_cunningham_slip_correction,
)
from particula.backend.taichi.particles.properties.ti_slip_correction_module import (
    ti_get_cunningham_slip_correction,
    kget_cunningham_slip_correction,
)

ti.init(arch=ti.cpu)


def test_ti_wrapper_matches_numpy():
    # scalar
    kn_scalar = 0.15
    npt.assert_allclose(
        ti_get_cunningham_slip_correction(kn_scalar),
        get_cunningham_slip_correction(kn_scalar),
        rtol=1e-7,
    )
    # array
    kn_array = np.linspace(0.05, 2.0, 8)
    npt.assert_allclose(
        ti_get_cunningham_slip_correction(kn_array),
        get_cunningham_slip_correction(kn_array),
        rtol=1e-7,
    )


def test_kernel_direct_matches_numpy():
    kn_array = np.linspace(0.05, 1.5, 6)
    kn_ti = ti.ndarray(dtype=ti.f64, shape=kn_array.size)
    res_ti = ti.ndarray(dtype=ti.f64, shape=kn_array.size)
    kn_ti.from_numpy(kn_array)

    kget_cunningham_slip_correction(kn_ti, res_ti)

    npt.assert_allclose(
        res_ti.to_numpy(),
        get_cunningham_slip_correction(kn_array),
        rtol=1e-7,
    )
