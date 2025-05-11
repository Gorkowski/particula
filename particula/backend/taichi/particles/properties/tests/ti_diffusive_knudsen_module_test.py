import taichi as ti, numpy as np
ti.init(arch=ti.cpu)

from particula.particles.properties.diffusive_knudsen_module import (
    get_diffusive_knudsen_number as ref_fn,
)
from particula.backend.taichi.particles.properties.ti_diffusive_knudsen_module import (
    ti_get_diffusive_knudsen_number,
    kget_diffusive_knudsen_number,
)

def test_wrapper_matches_numpy():
    r = np.array([1e-7, 2e-7])
    m = np.array([1e-17, 2e-17])
    f = np.array([0.8, 1.1])
    phi = np.array([0.1, 0.2])
    np.testing.assert_allclose(
        ti_get_diffusive_knudsen_number(r, m, f, phi),
        ref_fn(r, m, f, phi),
        rtol=1e-7,
    )

def test_kernel_direct():
    # small 1×1 case
    r = np.array([1e-7])
    m = np.array([1e-17])
    f = np.array([0.8])
    phi = 0.0
    # build auxiliaries exactly as wrapper
    ...
