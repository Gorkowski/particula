import taichi as ti
import numpy as np
import numpy.testing as npt

from particula.particles.properties.diffusive_knudsen_module \
    import get_diffusive_knudsen_number as ref_impl
from particula.backend.taichi.particles.properties.ti_diffusive_knudsen_module \
    import kget_diffusive_knudsen_number, ti_get_diffusive_knudsen_number

ti.init(arch=ti.cpu)

def test_ti_wrapper_against_numpy():
    r = np.array([1e-8, 2e-8, 3e-8])
    m = np.array([1e-20, 2e-20, 3e-20])
    f = np.array([0.5, 1.1, 2.0])
    cp= np.array([0.2, 0.3, 0.4])
    T = 310.0
    npt.assert_allclose(
        ti_get_diffusive_knudsen_number(r, m, f, cp, T),
        ref_impl(r, m, f, cp, T),
        rtol=1e-12, atol=0
    )

def test_kernel_direct():
    r = np.array([1e-7], dtype=np.float64)
    m = np.array([1e-17], dtype=np.float64)
    f = np.array([0.8], dtype=np.float64)
    cp= np.array([0.3], dtype=np.float64)
    T = 298.15
    out_ti = ti.ndarray(dtype=ti.f64, shape=(1, 1))
    kget_diffusive_knudsen_number(
        ti.ndarray(ti.f64, shape=1).from_numpy(r),
        ti.ndarray(ti.f64, shape=1).from_numpy(m),
        ti.ndarray(ti.f64, shape=1).from_numpy(f),
        ti.ndarray(ti.f64, shape=1).from_numpy(cp),
        T,
        out_ti,
    )
    npt.assert_allclose(
        out_ti.to_numpy()[0, 0],
        ref_impl(r.item(), m.item(), f.item(), cp.item(), T),
        rtol=1e-12, atol=0
    )
