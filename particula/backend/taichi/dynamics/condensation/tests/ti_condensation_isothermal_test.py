import numpy as np
import taichi as ti
ti.init(arch=ti.cpu, default_fp=ti.f64)

from particula.backend.taichi.dynamics.condensation.ti_condensation_strategies \
    import CondensationIsothermal as TiCondensationIsothermal
from particula.dynamics.condensation.condensation_strategies \
    import CondensationIsothermal as PyCondensationIsothermal


def test_first_order_mass_transport_parity():
    """Ensure Ti implementation matches NumPy reference."""
    particle_radius = np.array([1e-7, 2e-7, 5e-8], dtype=np.float64)
    temperature, pressure = 298.15, 101_325.0
    molar_mass = 0.018  # kg mol⁻¹

    ti_condensation_isothermal = TiCondensationIsothermal(molar_mass=molar_mass)
    py_condensation_isothermal = PyCondensationIsothermal(molar_mass=molar_mass)

    mass_transport_coefficient_ti = ti_condensation_isothermal.first_order_mass_transport(
        particle_radius, temperature, pressure
    )
    mass_transport_coefficient_py = py_condensation_isothermal.first_order_mass_transport(
        particle_radius, temperature, pressure
    )

    np.testing.assert_allclose(
        mass_transport_coefficient_ti, mass_transport_coefficient_py, rtol=1e-7
    )


def test_kernel_runs():
    """Smoke test: kernels compile & return finite results."""
    particle_radius = np.array([1e-7, 1e-7], dtype=np.float64)
    ti_condensation_isothermal = TiCondensationIsothermal(molar_mass=0.018)
    mass_transport_coefficient = ti_condensation_isothermal.first_order_mass_transport(
        particle_radius, 300.0, 101325.0
    )
    assert np.all(np.isfinite(mass_transport_coefficient)) and mass_transport_coefficient.shape == particle_radius.shape
