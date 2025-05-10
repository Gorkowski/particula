import taichi as ti
import numpy as np
import pytest

from particula.backend.taichi.particles.properties.ti_stokes_number_module import (
    ti_get_stokes_number,
    kget_stokes_number,
)
from particula.particles.properties.stokes_number import get_stokes_number

ti.init(arch=ti.cpu)


def test_get_stokes_number_taichi_matches_numpy():
    pit = np.array([1e-3, 2e-3, 3e-3])
    kt = np.array([2e-3, 2e-3, 1e-3])
    expected = get_stokes_number(pit, kt)
    result = ti_get_stokes_number(pit, kt)
    np.testing.assert_allclose(result, expected)


def test_kget_stokes_number_kernel_direct():
    pit = np.array([1e-3, 2e-3, 3e-3], dtype=np.float64)
    kt = np.array([2e-3, 2e-3, 1e-3], dtype=np.float64)
    expected = pit / kt
    pit_ti = ti.ndarray(dtype=ti.f64, shape=pit.size)
    kt_ti = ti.ndarray(dtype=ti.f64, shape=kt.size)
    res_ti = ti.ndarray(dtype=ti.f64, shape=pit.size)
    pit_ti.from_numpy(pit)
    kt_ti.from_numpy(kt)
    kget_stokes_number(pit_ti, kt_ti, res_ti)
    np.testing.assert_allclose(res_ti.to_numpy(), expected)
