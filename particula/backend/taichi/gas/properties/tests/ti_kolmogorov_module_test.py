import taichi as ti
ti.init(arch=ti.cpu)

import numpy as np
from particula.gas.properties.kolmogorov_module import (
    get_kolmogorov_time,
    get_kolmogorov_length,
    get_kolmogorov_velocity,
)
from particula.backend.taichi.gas.properties.ti_kolmogorov_module import (
    kget_kolmogorov_time,
    kget_kolmogorov_length,
    kget_kolmogorov_velocity,
    ti_get_kolmogorov_time,
    ti_get_kolmogorov_length,
    ti_get_kolmogorov_velocity,
)

def _sample_data():
    # Return two arrays of positive floats, length >= 2
    v = np.array([1.5e-5, 2.0e-5], dtype=np.float64)
    eps = np.array([0.1, 0.2], dtype=np.float64)
    return v, eps

def test_ti_wrappers_parity():
    v, eps = _sample_data()
    # Vector input
    np.testing.assert_allclose(
        ti_get_kolmogorov_time(v, eps),
        get_kolmogorov_time(v, eps),
        rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        ti_get_kolmogorov_length(v, eps),
        get_kolmogorov_length(v, eps),
        rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        ti_get_kolmogorov_velocity(v, eps),
        get_kolmogorov_velocity(v, eps),
        rtol=1e-12, atol=0
    )
    # Scalar input
    v0, eps0 = v[0], eps[0]
    np.testing.assert_allclose(
        ti_get_kolmogorov_time(v0, eps0),
        get_kolmogorov_time(v0, eps0),
        rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        ti_get_kolmogorov_length(v0, eps0),
        get_kolmogorov_length(v0, eps0),
        rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        ti_get_kolmogorov_velocity(v0, eps0),
        get_kolmogorov_velocity(v0, eps0),
        rtol=1e-12, atol=0
    )

def test_ti_kernels_parity():
    v, eps = _sample_data()
    n = v.size
    v_ti = ti.ndarray(dtype=ti.f64, shape=n)
    eps_ti = ti.ndarray(dtype=ti.f64, shape=n)
    out_time = ti.ndarray(dtype=ti.f64, shape=n)
    out_length = ti.ndarray(dtype=ti.f64, shape=n)
    out_velocity = ti.ndarray(dtype=ti.f64, shape=n)
    v_ti.from_numpy(v)
    eps_ti.from_numpy(eps)
    kget_kolmogorov_time(v_ti, eps_ti, out_time)
    kget_kolmogorov_length(v_ti, eps_ti, out_length)
    kget_kolmogorov_velocity(v_ti, eps_ti, out_velocity)
    np.testing.assert_allclose(
        out_time.to_numpy(),
        get_kolmogorov_time(v, eps),
        rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        out_length.to_numpy(),
        get_kolmogorov_length(v, eps),
        rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        out_velocity.to_numpy(),
        get_kolmogorov_velocity(v, eps),
        rtol=1e-12, atol=0
    )
