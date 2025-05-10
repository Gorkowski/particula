import numpy as np
import numpy.testing as npt
import taichi as ti

ti.init(arch=ti.cpu)

from particula.backend.taichi.particles.properties.ti_knudsen_number_module import (
    ti_get_knudsen_number,
    kget_knudsen_number,
)
from particula.particles.properties.knudsen_number_module import get_knudsen_number


def test_wrapper_matches_numpy():
    mfp = np.array([6.5e-8, 4.2e-8])
    radius = np.array([1.0e-7, 2.0e-7])
    npt.assert_allclose(
        ti_get_knudsen_number(mfp, radius),
        get_knudsen_number(mfp, radius),
    )


def test_kernel_direct():
    mfp = np.random.rand(10) + 1e-8
    radius = np.random.rand(10) + 1e-6
    ref = get_knudsen_number(mfp, radius)

    mfp_ti = ti.ndarray(dtype=ti.f64, shape=10)
    rad_ti = ti.ndarray(dtype=ti.f64, shape=10)
    res_ti = ti.ndarray(dtype=ti.f64, shape=10)
    mfp_ti.from_numpy(mfp)
    rad_ti.from_numpy(radius)

    kget_knudsen_number(mfp_ti, rad_ti, res_ti)
    npt.assert_allclose(res_ti.to_numpy(), ref)
