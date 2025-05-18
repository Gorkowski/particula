import unittest
import numpy as np
import taichi as ti
ti.init(arch=ti.cpu, default_fp=ti.f64)

from particula.backend.taichi.dynamics.condensation.ti_condensation_strategies import (
    TiCondensationIsothermal,
)
from particula.backend.taichi.particles.ti_distribution_strategies import (
    TiParticleResolvedSpeciatedMass,
)
from particula.backend.taichi.particles.ti_activity_strategies import (
    ActivityKappaParameter,
)
from particula.backend.taichi.particles.ti_surface_strategies import (
    SurfaceStrategyMass,
)
from particula.backend.taichi.particles.ti_representation import (
    TiParticleRepresentation,
)
from particula.backend.taichi.gas.ti_species import TiGasSpecies
from particula.backend.taichi.gas.ti_vapor_pressure_strategies import (
    WaterBuckStrategy,
    ConstantVaporPressureStrategy,
)

def _build_taichi_condensation_isothermal(
    molar_mass: np.ndarray,
    diffusion_coefficient: float = 2.0e-5,
    accommodation: float = 1.0,
):
    mm_ti = ti.ndarray(dtype=ti.f64, shape=molar_mass.shape)
    mm_ti.from_numpy(molar_mass)

    diff_coeff_ti = ti.field(ti.f64, shape=())
    diff_coeff_ti[None] = diffusion_coefficient

    alpha_ti = ti.ndarray(dtype=ti.f64, shape=molar_mass.shape)
    alpha_ti.from_numpy(np.full_like(molar_mass, accommodation))

    # let constructor run, then overwrite with the prepared fields
    cond = TiCondensationIsothermal(
        molar_mass=mm_ti,
        diffusion_coefficient=diff_coeff_ti,
        accommodation_coefficient=alpha_ti,
        update_gases=False,
    )
    cond.molar_mass = mm_ti
    cond.diffusion_coefficient = diff_coeff_ti
    cond.accommodation_coefficient = alpha_ti
    return cond

class TiCondensationIsothermalTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.n_particles = 10
        self.n_species = 3

        # ---- particle side ------------------------------------------------
        self.distribution = np.abs(
            np.random.randn(self.n_particles, self.n_species)
        ) * 1e-18                                 # kg of each species
        self.densities = np.array([1000., 1200., 1500.])
        self.concentration = np.ones(self.n_particles)
        self.charge = np.zeros(self.n_particles)

        strategy = TiParticleResolvedSpeciatedMass()
        activity = ActivityKappaParameter(
            kappa=np.array([0.5, 0.0, 0.0]),
            density=self.densities,
            molar_mass=np.array([0.018, 0.050, 0.040]),
            water_index=0,
        )
        surface = SurfaceStrategyMass(
            surface_tension=0.072,
            density=self.densities,
        )

        self.particle = TiParticleRepresentation(
            strategy,
            activity,
            surface,
            self.distribution,
            self.densities,
            self.concentration,
            self.charge,
            volume=1.0,
        )

        # ---- gas side -----------------------------------------------------
        self.gas = TiGasSpecies(
            name=np.array(["H2O", "X1", "X2"]),
            molar_mass=np.array([0.018, 0.050, 0.040]),
            vapor_pressure_strategy=[
                WaterBuckStrategy(),
                ConstantVaporPressureStrategy(100.0),
                ConstantVaporPressureStrategy(200.0),
            ],
            partitioning=True,
            concentration=np.array([1.0, 1.0, 1.0]),
        )

        # ---- condensation object -----------------------------------------
        self.condensation = _build_taichi_condensation_isothermal(
            molar_mass=np.array([0.018, 0.050, 0.040])
        )

        self.temperature = 298.15
        self.pressure = 101_325.0
        self.dynamic_visc = 1.85e-5

    # ---------------------------------------------------------------------
    def test_first_order_mass_transport_shape_and_finite(self):
        radius = self.particle.get_radius().to_numpy()
        coeff = self.condensation.first_order_mass_transport(
            radius,
            self.temperature,
            self.pressure,
            self.dynamic_visc,
        )
        self.assertEqual(coeff.shape, (self.n_particles, self.n_species))
        self.assertTrue(np.all(np.isfinite(coeff.to_numpy())))

    def test_mass_transfer_rate_shape_and_finite(self):
        dm_dt = self.condensation.mass_transfer_rate(
            particle=self.particle,
            gas_species=self.gas,
            temperature=self.temperature,
            pressure=self.pressure,
            dynamic_viscosity=self.dynamic_visc,
        )
        self.assertEqual(dm_dt.shape, (self.n_particles, self.n_species))
        self.assertTrue(np.all(np.isfinite(dm_dt.to_numpy())))

