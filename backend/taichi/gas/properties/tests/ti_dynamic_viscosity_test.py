import numpy as np
import taichi as ti
import particula as par
from particula.backend.taichi.gas.properties.ti_dynamic_viscosity_module import (
    kget_dynamic_viscosity,
    fget_dynamic_viscosity,
)

ti.init(arch=ti.cpu)

def test_wrapper_vs_numpy():
    T = np.array([250.0, 300.0, 350.0])
    ref_visc = par.util.constants.REF_VISCOSITY_AIR_STP
    ref_temp = par.util.constants.REF_TEMPERATURE_STP

    expected = par.gas.properties.dynamic_viscosity.get_dynamic_viscosity(T)
    result = par.gas.properties.dynamic_viscosity.get_dynamic_viscosity(
        T, backend="taichi"
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)

def test_kernel_direct_call():
    T = np.array([280.0, 300.0])
    ref_visc = np.full_like(T, par.util.constants.REF_VISCOSITY_AIR_STP)
    ref_temp = np.full_like(T, par.util.constants.REF_TEMPERATURE_STP)
    res = np.empty_like(T)

    t_ti  = ti.ndarray(dtype=ti.f64, shape=T.size);  t_ti.from_numpy(T)
    rv_ti = ti.ndarray(dtype=ti.f64, shape=T.size);  rv_ti.from_numpy(ref_visc)
    rt_ti = ti.ndarray(dtype=ti.f64, shape=T.size);  rt_ti.from_numpy(ref_temp)
    res_ti = ti.ndarray(dtype=ti.f64, shape=T.size)

    kget_dynamic_viscosity(t_ti, rv_ti, rt_ti, res_ti)
    np.testing.assert_allclose(res_ti.to_numpy(), res := (
        par.gas.properties.dynamic_viscosity.get_dynamic_viscosity(T)
    ), rtol=1e-12, atol=0)
