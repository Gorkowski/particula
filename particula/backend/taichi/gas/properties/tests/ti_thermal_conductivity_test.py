import taichi as ti
import numpy as np
import numpy.testing as npt

from particula.gas.properties.thermal_conductivity import get_thermal_conductivity
from particula.backend.taichi.gas.properties.ti_thermal_conductivity_module import (
    kget_thermal_conductivity, ti_get_thermal_conductivity
)

ti.init(arch=ti.cpu)

def test_wrapper_matches_numpy():
    temp = np.linspace(250.0, 350.0, 11)
    npt.assert_allclose(
        ti_get_thermal_conductivity(temp),
        get_thermal_conductivity(temp),
        rtol=1e-13, atol=0
    )

def test_kernel_direct():
    temp = np.array([280.0, 300.0, 320.0], dtype=np.float64)
    res_ti = ti.ndarray(dtype=ti.f64, shape=temp.size)
    temp_ti = ti.ndarray(dtype=ti.f64, shape=temp.size)
    temp_ti.from_numpy(temp)
    kget_thermal_conductivity(temp_ti, res_ti)
    npt.assert_allclose(
        res_ti.to_numpy(),
        get_thermal_conductivity(temp),
        rtol=1e-13, atol=0
    )
