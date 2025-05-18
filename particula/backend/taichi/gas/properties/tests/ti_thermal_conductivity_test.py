import taichi as ti
import numpy as np
import numpy.testing as npt

from particula.gas.properties.thermal_conductivity import get_thermal_conductivity
from particula.backend.taichi.gas.properties.ti_thermal_conductivity_module import (
    kget_thermal_conductivity, ti_get_thermal_conductivity
)

ti.init(arch=ti.cpu)

def test_wrapper_matches_numpy():
    temperature_array = np.linspace(250.0, 350.0, 11)
    npt.assert_allclose(
        ti_get_thermal_conductivity(temperature_array),
        get_thermal_conductivity(temperature_array),
        rtol=1e-7, atol=0
    )

def test_kernel_direct():
    temperature_array = np.array([280.0, 300.0, 320.0], dtype=np.float64)
    thermal_conductivity_taichi = ti.ndarray(
        dtype=ti.f64, shape=temperature_array.size
    )
    temperature_taichi = ti.ndarray(
        dtype=ti.f64, shape=temperature_array.size
    )
    temperature_taichi.from_numpy(temperature_array)
    kget_thermal_conductivity(
        temperature_taichi, thermal_conductivity_taichi
    )
    npt.assert_allclose(
        thermal_conductivity_taichi.to_numpy(),
        get_thermal_conductivity(temperature_array),
        rtol=1e-7, atol=0
    )
