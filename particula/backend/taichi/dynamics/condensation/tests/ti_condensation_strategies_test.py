import taichi as ti; ti.init(arch=ti.cpu, default_fp=ti.f64)
import numpy as np
from particula.backend.taichi.dynamics.condensation.ti_condensation_strategies import (
    TiCondensationIsothermal,
)

class DummyParticle:
    def __init__(self, radius, concentration, species_mass):
        self._radius = np.array(radius, dtype=np.float64)
        self.concentration = np.array(concentration, dtype=np.float64)
        self._species_mass = np.array(species_mass, dtype=np.float64)
    def get_radius(self):
        return self._radius
    def get_species_mass(self):
        return self._species_mass
    def get_concentration(self):
        return self.concentration
    def add_mass(self, added_mass):
        self._species_mass += added_mass

class DummyGas:
    def __init__(self, pure_vapor_pressure, partial_pressure, concentration):
        self._pure_vapor_pressure = pure_vapor_pressure
        self._partial_pressure = partial_pressure
        self._concentration = concentration
    def get_pure_vapor_pressure(self, temperature):
        return self._pure_vapor_pressure
    def get_partial_pressure(self, temperature):
        return self._partial_pressure
    def get_concentration(self):
        return self._concentration
    def add_concentration(self, added_concentration):
        self._concentration += added_concentration

class DummyActivity:
    def partial_pressure(self, pure_vapor_pressure, mass_concentration):
        return pure_vapor_pressure * 0.9

class DummySurface:
    def kelvin_term(self, radius, molar_mass, mass_concentration, temperature):
        return 1.0

# Patch dummy objects into DummyParticle
DummyParticle.activity = DummyActivity()
DummyParticle.surface = DummySurface()

def test_ti_condensation_isothermal_numerical_parity():
    # Minimal test system: 3 particles, 1 gas
    radius = [1e-7, 2e-7, 3e-7]
    concentration = [100, 200, 300]
    species_mass = [1e-15, 2e-15, 3e-15]
    pure_vapor_pressure = 100.0
    partial_pressure = 90.0
    gas_concentration = 1.0

    particle = DummyParticle(radius, concentration, species_mass)
    gas = DummyGas(pure_vapor_pressure, partial_pressure, gas_concentration)

    # Use the same parameters as the reference
    molar_mass = 0.018
    diffusion_coefficient = 2e-5
    accommodation_coefficient = 1.0

    ti_impl = TiCondensationIsothermal(
        molar_mass=molar_mass,
        diffusion_coefficient=diffusion_coefficient,
        accommodation_coefficient=accommodation_coefficient,
    )

    # Reference: use the original NumPy implementation
    from particula.dynamics.condensation.condensation_strategies import CondensationIsothermal
    ref_impl = CondensationIsothermal(
        molar_mass=molar_mass,
        diffusion_coefficient=diffusion_coefficient,
        accommodation_coefficient=accommodation_coefficient,
    )

    T = 298.15
    P = 101325.0

    ti_out = ti_impl.mass_transfer_rate(particle, gas, T, P)
    ref = ref_impl.mass_transfer_rate(particle, gas, T, P)

    assert ti_out.shape == ref.shape
    assert np.all(np.isfinite(ti_out))
    np.testing.assert_allclose(ref, ti_out, rtol=1e-12)
