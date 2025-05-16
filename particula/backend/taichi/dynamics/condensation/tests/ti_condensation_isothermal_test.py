import numpy as np
import taichi as ti
ti.init(arch=ti.cpu, default_fp=ti.f64)

from particula.backend.taichi.dynamics.condensation.ti_condensation_strategies \
    import CondensationIsothermal as TiCondensationIsothermal
from particula.dynamics.condensation.condensation_strategies \
    import CondensationIsothermal as PyCondensationIsothermal


def test_first_order_mass_transport_parity():
    """Ensure Ti implementation matches NumPy reference."""
    radius = np.array([1e-7, 2e-7, 5e-8], dtype=np.float64)
    temperature, pressure = 298.15, 101_325.0
    molar_mass = 0.018  # kg mol⁻¹

    ti_impl = TiCondensationIsothermal(molar_mass=molar_mass)
    py_impl = PyCondensationIsothermal(molar_mass=molar_mass)

    k_ti = ti_impl.first_order_mass_transport(radius, temperature, pressure)
    k_py = py_impl.first_order_mass_transport(radius, temperature, pressure)

    np.testing.assert_allclose(k_ti, k_py, rtol=1e-12)


def test_kernel_runs():
    """Smoke test: kernels compile & return finite results."""
    radius = np.array([1e-7, 1e-7], dtype=np.float64)
    ti_impl = TiCondensationIsothermal(molar_mass=0.018)
    k = ti_impl.first_order_mass_transport(radius, 300.0, 101325.0)
    assert np.all(np.isfinite(k)) and k.shape == radius.shape
