import taichi as ti, numpy as np, numpy.testing as npt
ti.init(arch=ti.cpu)

from particula.dynamics.condensation.mass_transfer import (
    get_first_order_mass_transport_k as np_fk,
    get_mass_transfer_rate as np_mr,
    get_radius_transfer_rate as np_rr,
)
from particula.backend.taichi.dynamics.condensation.ti_mass_transfer_module import (
    ti_get_first_order_mass_transport_k as ti_fk,
    kget_first_order_mass_transport_k,
    ti_get_mass_transfer_rate as ti_mr,
    kget_mass_transfer_rate,
    ti_get_radius_transfer_rate as ti_rr,
    kget_radius_transfer_rate,
)

def test_wrapper_functions():
    r = np.array([1e-6, 2e-6], dtype=np.float64)
    vt = np.array([0.6, 0.7], dtype=np.float64)
    d  = np.array([2e-5, 2e-5], dtype=np.float64)
    dp = np.array([10.0, 15.0], dtype=np.float64)
    k  = np_fk(r, vt, d)
    T  = 300.0
    m  = 0.02897
    dm = np_mr(dp, k, T, m)

    npt.assert_allclose(ti_fk(r, vt, d), k)
    npt.assert_allclose(ti_mr(dp, k, T, m), dm)
    npt.assert_allclose(ti_rr(dm, r, 1000.0),
                        np_rr(dm, r, 1000.0))

def test_kernels_direct():
    r = np.array([1e-6, 2e-6], dtype=np.float64)
    vt = np.array([0.6, 0.7], dtype=np.float64)
    d  = np.array([2e-5, 2e-5], dtype=np.float64)
    n  = r.size
    r_ti, vt_ti, d_ti = [ti.ndarray(ti.f64, n) for _ in range(3)]
    res_ti = ti.ndarray(ti.f64, n)
    r_ti.from_numpy(r); vt_ti.from_numpy(vt); d_ti.from_numpy(d)
    kget_first_order_mass_transport_k(r_ti, vt_ti, d_ti, res_ti)
    npt.assert_allclose(res_ti.to_numpy(), np_fk(r, vt, d))

    # repeat for the other two kernels
