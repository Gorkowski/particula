import numpy as np
import taichi as ti

ti.init(arch=ti.cpu)

from particula.backend.taichi.gas.properties.ti_dynamic_viscosity_module import (
    ti_get_dynamic_viscosity,
    kget_dynamic_viscosity,
)
from particula.gas.properties.dynamic_viscosity import get_dynamic_viscosity
from particula.util.constants import (
    REF_TEMPERATURE_STP,
    REF_VISCOSITY_AIR_STP,
)

def test_wrapper_matches_reference():
    temps = np.array([250.0, 300.0, 350.0])
    np.testing.assert_allclose(
        ti_get_dynamic_viscosity(temps),
        get_dynamic_viscosity(temps),
        rtol=1e-12,
    )

def test_kernel_direct():
    temps = np.array([280.0, 310.0], dtype=np.float64)
    n = temps.size
    temps_ti = ti.ndarray(dtype=ti.f64, shape=n)
    rv_ti    = ti.ndarray(dtype=ti.f64, shape=n)
    rt_ti    = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti   = ti.ndarray(dtype=ti.f64, shape=n)

    temps_ti.from_numpy(temps)
    rv_ti.from_numpy(np.full(n, REF_VISCOSITY_AIR_STP, dtype=np.float64))
    rt_ti.from_numpy(np.full(n, REF_TEMPERATURE_STP, dtype=np.float64))

    kget_dynamic_viscosity(temps_ti, rv_ti, rt_ti, res_ti)

    np.testing.assert_allclose(
        res_ti.to_numpy(),
        get_dynamic_viscosity(temps),
        rtol=1e-12,
    )
