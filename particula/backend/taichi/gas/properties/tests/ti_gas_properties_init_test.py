import numpy as np
import taichi as ti
ti.init(arch=ti.cpu)

from particula.backend.taichi.gas.properties import (
    ti_get_dynamic_viscosity,
)

from particula.gas.properties.dynamic_viscosity import get_dynamic_viscosity
from particula.backend.taichi.gas.properties.ti_dynamic_viscosity_module import (
    kget_dynamic_viscosity,
)

def test_imports_are_callable():
    assert callable(ti_get_dynamic_viscosity)

def test_wrapper_matches_python():
    t   = np.array([288.15, 300.0], dtype=np.float64)
    mu0 = np.full_like(t, 1.827e-5)
    T0  = np.full_like(t, 288.15)

    np.testing.assert_allclose(
        ti_get_dynamic_viscosity(t, mu0, T0),
        get_dynamic_viscosity(t, reference_viscosity=mu0, reference_temperature=T0),
    )

def test_kernel_direct_call():
    t   = np.array([288.15, 300.0], dtype=np.float64)
    mu0 = np.full_like(t, 1.827e-5)
    T0  = np.full_like(t, 288.15)
    n   = t.size

    t_ti   = ti.ndarray(dtype=ti.f64, shape=n); t_ti.from_numpy(t)
    mu0_ti = ti.ndarray(dtype=ti.f64, shape=n); mu0_ti.from_numpy(mu0)
    T0_ti  = ti.ndarray(dtype=ti.f64, shape=n); T0_ti.from_numpy(T0)
    out_ti = ti.ndarray(dtype=ti.f64, shape=n)

    kget_dynamic_viscosity(t_ti, mu0_ti, T0_ti, out_ti)

    np.testing.assert_allclose(
        out_ti.to_numpy(),
        get_dynamic_viscosity(t, reference_viscosity=mu0, reference_temperature=T0),
    )
