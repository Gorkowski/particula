import taichi as ti, numpy as np, numpy.testing as npt
ti.init(arch=ti.cpu)

# python (NumPy) reference implementation
from particula.dynamics.condensation.mass_transfer import (
    get_first_order_mass_transport_via_system_state as np_fk_vs,
)

# taichi wrapper + kernel
from particula.backend.taichi.dynamics.condensation.ti_mass_transfer import (
    ti_get_first_order_mass_transport_via_system_state as ti_fk_vs,
    kget_first_order_mass_transport_via_system_state,
)

def _example_inputs():
    r    = np.array([1e-6, 2e-6], dtype=np.float64)      # particle radii
    mm   = np.array([0.02897, 0.04401], dtype=np.float64)  # species molar masses
    ac   = np.array([1.0, 1.0], dtype=np.float64)        # accommodation coeffs
    T    = 300.0
    P    = 101325.0
    mu   = 1.8e-5
    D    = 2e-5
    return r, mm, ac, T, P, mu, D

def test_wrapper():
    r, mm, ac, T, P, mu, D = _example_inputs()
    expected = np_fk_vs(r, mm, ac, T, P, mu, D)
    result   = ti_fk_vs(r, mm, ac, T, P, mu, D)
    npt.assert_allclose(result, expected)

def test_kernel_direct():
    r, mm, ac, T, P, mu, D = _example_inputs()
    n_p, n_s = r.size, mm.size
    r_ti   = ti.ndarray(ti.f64, n_p)
    mm_ti  = ti.ndarray(ti.f64, n_s)
    ac_ti  = ti.ndarray(ti.f64, n_p)
    res_ti = ti.ndarray(ti.f64, (n_p, n_s))
    r_ti.from_numpy(r)
    mm_ti.from_numpy(mm)
    ac_ti.from_numpy(ac)
    kget_first_order_mass_transport_via_system_state(
        r_ti, mm_ti, ac_ti, T, P, mu, D, res_ti
    )
    expected = np_fk_vs(r, mm, ac, T, P, mu, D)
    npt.assert_allclose(res_ti.to_numpy(), expected)
