import numpy as np
import taichi as ti

ti.init(arch=ti.cpu)

import particula as par
from particula.backend.taichi.particles.properties.ti_mixing_state_index_module import (
    kget_mixing_state_index,
    ti_get_mixing_state_index,
)


def reference(m):
    return par.particles.get_mixing_state_index(m)


def test_wrapper():
    m = np.array([[1e-15, 0.0], [5e-16, 5e-16]])
    np.testing.assert_allclose(
        ti_get_mixing_state_index(m), reference(m), rtol=1e6
    )


def test_kernel_direct():
    m = np.random.rand(3, 4)  # 3 particles, 4 species
    n_p, n_s = m.shape
    m_ti = ti.ndarray(dtype=ti.f64, shape=m.shape)
    m_ti.from_numpy(m)
    out_ti = ti.ndarray(dtype=ti.f64, shape=1)
    kget_mixing_state_index(m_ti, n_p, n_s, out_ti)
    np.testing.assert_allclose(out_ti.to_numpy()[0], reference(m), rtol=1e6)
