import numpy as np
import taichi as ti
import particula as par
from particula.backend.taichi.gas.properties.ti_taylor_microscale_module import (
    kget_lagrangian_taylor_microscale_time,
    kget_taylor_microscale,
    kget_taylor_microscale_reynolds_number,
    ti_get_lagrangian_taylor_microscale_time,
    ti_get_taylor_microscale,
    ti_get_taylor_microscale_reynolds_number,
)

ti.init(arch=ti.cpu)

def test_wrappers_vs_numpy():
    # vector inputs
    kt  = np.array([0.38, 0.40])
    rl  = np.array([400., 600.])
    av  = np.array([0.05, 0.06])
    u   = np.array([0.35, 0.40])
    nu  = np.array([1.5e-5, 1.8e-5])
    eps = np.array([0.10, 0.12])
    lam = par.gas.get_taylor_microscale(u, nu, eps)

    np.testing.assert_allclose(
        ti_get_lagrangian_taylor_microscale_time(kt, rl, av),
        par.gas.get_lagrangian_taylor_microscale_time(kt, rl, av),
    )
    np.testing.assert_allclose(
        ti_get_taylor_microscale(u, nu, eps),
        lam,
    )
    np.testing.assert_allclose(
        ti_get_taylor_microscale_reynolds_number(u, lam, nu),
        par.gas.get_taylor_microscale_reynolds_number(u, lam, nu),
    )

def test_kernels_direct():
    n = 3
    kt, rl, av = [np.full(n, v) for v in (0.40, 500., 0.05)]
    u, nu, eps = [np.full(n, v) for v in (0.35, 1.5e-5, 0.10)]
    lam = par.gas.get_taylor_microscale(u, nu, eps)

    res_ti = ti.ndarray(dtype=ti.f64, shape=n)

    # lagrangian time kernel
    kt_ti = ti.ndarray(ti.f64, n); rl_ti = ti.ndarray(ti.f64, n); av_ti = ti.ndarray(ti.f64, n)
    kt_ti.from_numpy(kt); rl_ti.from_numpy(rl); av_ti.from_numpy(av)
    kget_lagrangian_taylor_microscale_time(kt_ti, rl_ti, av_ti, res_ti)
    np.testing.assert_allclose(res_ti.to_numpy(),
        par.gas.get_lagrangian_taylor_microscale_time(kt, rl, av))

    # taylor microscale kernel
    u_ti = ti.ndarray(ti.f64, n); nu_ti = ti.ndarray(ti.f64, n); eps_ti = ti.ndarray(ti.f64, n)
    u_ti.from_numpy(u); nu_ti.from_numpy(nu); eps_ti.from_numpy(eps)
    kget_taylor_microscale(u_ti, nu_ti, eps_ti, res_ti)
    np.testing.assert_allclose(res_ti.to_numpy(), lam)

    # Reynolds number kernel
    lam_ti = ti.ndarray(ti.f64, n); lam_ti.from_numpy(lam)
    kget_taylor_microscale_reynolds_number(u_ti, lam_ti, nu_ti, res_ti)
    np.testing.assert_allclose(
        res_ti.to_numpy(),
        par.gas.get_taylor_microscale_reynolds_number(u, lam, nu),
    )
