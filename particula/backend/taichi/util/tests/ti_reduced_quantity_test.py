"""
Unit tests for the Taichi backend of the reduced quantity module.
"""
import taichi as ti
import numpy as np
import numpy.testing as npt

from particula.util.reduced_quantity import (
    get_reduced_value,
    get_reduced_self_broadcast,
)
from particula.backend.taichi.util.ti_reduced_quantity import (
    ti_get_reduced_value,
    ti_get_reduced_self_broadcast,
    kget_reduced_value,
    kget_reduced_self_broadcast,
)

ti.init(arch=ti.cpu)


def test_wrapper_elementwise():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 5.0, 10.0])
    npt.assert_allclose(
        ti_get_reduced_value(a, b),
        get_reduced_value(a, b),
    )


def test_wrapper_self_broadcast():
    a = np.array([1.0, 2.0, 3.0])
    npt.assert_allclose(
        ti_get_reduced_self_broadcast(a),
        get_reduced_self_broadcast(a),
    )

def test_kernel_elementwise_direct():
    a = np.array([1.0, 2.0])
    b = np.array([2.0, 4.0])
    out_ti = ti.ndarray(dtype=ti.f64, shape=a.size)
    a_ti   = ti.ndarray(dtype=ti.f64, shape=a.size)
    b_ti   = ti.ndarray(dtype=ti.f64, shape=b.size)
    a_ti.from_numpy(a); b_ti.from_numpy(b)
    kget_reduced_value(a_ti, b_ti, out_ti)
    npt.assert_allclose(out_ti.to_numpy(), get_reduced_value(a, b))


def test_kernel_self_broadcast_direct():
    a = np.array([0.0, 1.0])
    n = a.size
    out_ti  = ti.ndarray(dtype=ti.f64, shape=(n, n))
    a_ti    = ti.ndarray(dtype=ti.f64, shape=n)
    a_ti.from_numpy(a)
    kget_reduced_self_broadcast(a_ti, out_ti)
    npt.assert_allclose(out_ti.to_numpy(), get_reduced_self_broadcast(a))
