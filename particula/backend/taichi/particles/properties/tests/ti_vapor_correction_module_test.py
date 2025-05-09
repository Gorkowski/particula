import numpy as np
import taichi as ti
ti.init(arch=ti.cpu)

from particula.particles.properties.vapor_correction_module import (
    get_vapor_transition_correction as ref_func,
)
from particula.backend.taichi.particles.properties.ti_vapor_correction_module import (
    get_vapor_transition_correction_taichi,
    kget_vapor_transition_correction,
)

# ---------------------------------------------------------------------------
def test_get_vapor_transition_correction_taichi_array():
    kn   = np.linspace(0.0, 10.0, 50)
    alpha = np.full_like(kn, 0.9)
    np.testing.assert_allclose(
        get_vapor_transition_correction_taichi(kn, alpha),
        ref_func(knudsen_number=kn, mass_accommodation=alpha),
        rtol=1e-13, atol=0
    )

# ---------------------------------------------------------------------------
def test_kget_vapor_transition_correction_kernel_direct_call():
    kn   = np.array([0.0, 0.1, 1.0, 2.5])
    alpha = np.array([1.0, 0.8, 0.5, 0.9])
    n = kn.size

    kn_ti  = ti.ndarray(dtype=ti.f64, shape=n)
    al_ti  = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    kn_ti.from_numpy(kn)
    al_ti.from_numpy(alpha)

    kget_vapor_transition_correction(kn_ti, al_ti, res_ti)
    np.testing.assert_allclose(
        res_ti.to_numpy(),
        ref_func(knudsen_number=kn, mass_accommodation=alpha),
        rtol=1e-13, atol=0
    )
