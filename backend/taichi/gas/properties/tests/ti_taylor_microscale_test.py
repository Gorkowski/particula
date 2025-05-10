import numpy as np
import taichi as ti
import particula as par
from particula.backend.taichi.gas.properties.ti_taylor_microscale_module import (
    kget_lagrangian_taylor_microscale_time,
)

ti.init(arch=ti.cpu)

def test_wrapper_matches_python():
    kol = np.array([0.3, 0.5])
    re_lam = np.array([200.0, 500.0])
    acc = np.array([0.05, 0.08])

    expected = par.gas.get_lagrangian_taylor_microscale_time(kol, re_lam, acc)

    par.backend.use_backend("taichi")
    got = par.gas.get_lagrangian_taylor_microscale_time(kol, re_lam, acc)
    par.backend.use_backend("python")

    np.testing.assert_allclose(got, expected)

def test_kernel_direct():
    kol = np.array([0.2, 0.4])
    re_lam = np.array([150.0, 350.0])
    acc = np.array([0.06, 0.07])

    kol_ti = ti.ndarray(dtype=ti.f64, shape=kol.size); kol_ti.from_numpy(kol)
    re_ti = ti.ndarray(dtype=ti.f64, shape=re_lam.size); re_ti.from_numpy(re_lam)
    acc_ti = ti.ndarray(dtype=ti.f64, shape=acc.size);   acc_ti.from_numpy(acc)
    out_ti = ti.ndarray(dtype=ti.f64, shape=kol.size)

    kget_lagrangian_taylor_microscale_time(kol_ti, re_ti, acc_ti, out_ti)

    expected = kol * np.sqrt((2 * re_lam) / (np.sqrt(15) * acc))
    np.testing.assert_allclose(out_ti.to_numpy(), expected)
