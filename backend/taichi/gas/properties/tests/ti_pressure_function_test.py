import numpy as np
import taichi as ti
ti.init(arch=ti.cpu)

from particula.gas.properties.pressure_function import (
    get_partial_pressure,
    get_saturation_ratio_from_pressure,
)

def test_wrapper_agrees_with_numpy():
    c  = np.array([1.2, 0.5, 3.4])
    m  = np.array([0.02897, 0.044, 0.02])
    T  = np.array([298.0, 260.0, 310.0])
    p_ref = (c * 8.314462618 * T) / m
    np.testing.assert_allclose(
        get_partial_pressure(c, m, T), p_ref
    )

    pvap = np.array([950., 800., 1000.])
    S_ref = p_ref / pvap
    np.testing.assert_allclose(
        get_saturation_ratio_from_pressure(p_ref, pvap), S_ref
    )

# Additional test: call kernels directly and compare
from particula.backend.taichi.gas.properties.ti_pressure_function_module import (
    kget_partial_pressure,
    kget_saturation_ratio,
    fget_partial_pressure,
    fget_saturation_ratio,
)

def test_kernels_direct():
    c  = np.array([1.2, 0.5, 3.4], dtype=np.float64)
    m  = np.array([0.02897, 0.044, 0.02], dtype=np.float64)
    T  = np.array([298.0, 260.0, 310.0], dtype=np.float64)
    out = np.zeros_like(c)
    c_ti = ti.ndarray(dtype=ti.f64, shape=c.shape); c_ti.from_numpy(c)
    m_ti = ti.ndarray(dtype=ti.f64, shape=m.shape); m_ti.from_numpy(m)
    T_ti = ti.ndarray(dtype=ti.f64, shape=T.shape); T_ti.from_numpy(T)
    out_ti = ti.ndarray(dtype=ti.f64, shape=out.shape)
    kget_partial_pressure(c_ti, m_ti, T_ti, out_ti)
    np.testing.assert_allclose(out_ti.to_numpy(), (c * 8.314462618 * T) / m)

    p = (c * 8.314462618 * T) / m
    pvap = np.array([950., 800., 1000.], dtype=np.float64)
    out2 = np.zeros_like(p)
    p_ti = ti.ndarray(dtype=ti.f64, shape=p.shape); p_ti.from_numpy(p)
    pvap_ti = ti.ndarray(dtype=ti.f64, shape=pvap.shape); pvap_ti.from_numpy(pvap)
    out2_ti = ti.ndarray(dtype=ti.f64, shape=out2.shape)
    kget_saturation_ratio(p_ti, pvap_ti, out2_ti)
    np.testing.assert_allclose(out2_ti.to_numpy(), p / pvap)
