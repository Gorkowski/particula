import numpy as np
import pytest
import taichi as ti

from particula.backend.taichi.util.ti_field_helper import FieldIO

def test_fieldio_from_numpy_1d():
    src = np.random.rand(10).astype(np.float64)
    dst = ti.field(dtype=ti.f64, shape=src.shape[0])
    FieldIO().from_numpy(dst, src)
    np.testing.assert_allclose(dst.to_numpy(), src)

def test_fieldio_from_numpy_2d():
    src = np.random.rand(3, 4).astype(np.float64)
    dst = ti.field(dtype=ti.f64, shape=src.shape)
    FieldIO().from_numpy(dst, src)
    np.testing.assert_allclose(dst.to_numpy(), src)

def test_fieldio_from_numpy_invalid_dimension():
    src = np.zeros((2, 2, 2), dtype=np.float64)   # 3-D array
    dst = ti.field(dtype=ti.f64, shape=8)          # any 1-D field
    with pytest.raises(ValueError):
        FieldIO().from_numpy(dst, src)
