import taichi as ti
ti.init(arch=ti.cpu)
import numpy as np
from numpy.testing import assert_allclose

from particula.backend.taichi.gas.properties.ti_fluid_rms_velocity_module import (
    get_fluid_rms_velocity_taichi, kget_fluid_rms_velocity
)
from particula.gas.properties.fluid_rms_velocity import get_fluid_rms_velocity

def test_ti_get_fluid_rms_velocity_wrapper_agreement():
    rl = np.array([500.0, 600.0])
    kv = np.array([1.5e-5, 1.7e-5])
    td = np.array([0.1, 0.12])
    expected = get_fluid_rms_velocity(rl, kv, td)
    actual = get_fluid_rms_velocity_taichi(rl, kv, td)
    assert_allclose(actual, expected, rtol=1e-12, atol=0)

def test_kget_fluid_rms_velocity_kernel_direct():
    rl = np.array([500.0])
    kv = np.array([1.5e-5])
    td = np.array([0.1])
    expected = get_fluid_rms_velocity(rl, kv, td)

    rl_ti = ti.ndarray(dtype=ti.f64, shape=1)
    kv_ti = ti.ndarray(dtype=ti.f64, shape=1)
    td_ti = ti.ndarray(dtype=ti.f64, shape=1)
    result_ti = ti.ndarray(dtype=ti.f64, shape=1)

    rl_ti.from_numpy(rl)
    kv_ti.from_numpy(kv)
    td_ti.from_numpy(td)

    kget_fluid_rms_velocity(rl_ti, kv_ti, td_ti, result_ti)
    actual = result_ti.to_numpy()
    assert_allclose(actual, expected, rtol=1e-12, atol=0)
