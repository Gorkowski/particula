import taichi as ti
import numpy as np
import numpy.testing as npt

from particula.particles.properties.convert_mole_fraction import (
    get_mass_fractions_from_moles as np_impl,
)
from particula.backend.taichi.particles.properties.ti_convert_mole_fraction_module import (
    ti_get_mass_fractions_from_moles,
    kget_mass_fractions_1d,
    kget_mass_fractions_2d,
)

ti.init(arch=ti.cpu)


def test_wrapper_1d():
    mole = np.random.rand(7)
    mole /= mole.sum()
    mw = np.random.uniform(10.0, 200.0, 7)
    npt.assert_allclose(
        ti_get_mass_fractions_from_moles(mole, mw),
        np_impl(mole, mw),
        rtol=1e-12,
    )


def test_wrapper_2d():
    mole = np.random.rand(4, 5)
    mole /= mole.sum(axis=1, keepdims=True)
    mw = np.random.uniform(10.0, 200.0, 5)
    npt.assert_allclose(
        ti_get_mass_fractions_from_moles(mole, mw),
        np_impl(mole, mw),
        rtol=1e-12,
    )


def test_kernel_direct_1d():
    mole = np.array([0.2, 0.5, 0.3], dtype=np.float64)
    mw = np.array([18.0, 44.0, 28.0], dtype=np.float64)
    out_ti = ti.ndarray(ti.f64, shape=3)
    mole_ti, mw_ti = ti.ndarray(ti.f64, 3), ti.ndarray(ti.f64, 3)
    mole_ti.from_numpy(mole)
    mw_ti.from_numpy(mw)
    kget_mass_fractions_1d(mole_ti, mw_ti, out_ti)
    npt.assert_allclose(out_ti.to_numpy(), np_impl(mole, mw))


def test_kernel_direct_2d():
    mole = np.array([[0.2, 0.5, 0.3], [0.3, 0.3, 0.4]], dtype=np.float64)
    mw = np.array([18.0, 44.0, 28.0], dtype=np.float64)
    rows, cols = mole.shape
    out_ti = ti.ndarray(ti.f64, shape=(rows, cols))
    mole_ti = ti.ndarray(ti.f64, shape=(rows, cols))
    mw_ti = ti.ndarray(ti.f64, shape=cols)
    mole_ti.from_numpy(mole)
    mw_ti.from_numpy(mw)
    kget_mass_fractions_2d(mole_ti, mw_ti, out_ti)
    npt.assert_allclose(out_ti.to_numpy(), np_impl(mole, mw))
