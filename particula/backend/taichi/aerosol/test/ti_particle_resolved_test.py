import unittest
import taichi as ti
import numpy as np

# import the simulation module (already initialises fields/data)
from particula.backend.taichi.aerosol import ti_particle_resolved as sim

@ti.kernel
def k_calculate_radius():
    for p in ti.ndrange(sim.particle_count):
        sim.update_radius(p)

@ti.kernel
def k_calculate_first_order_coefficient():
    for p in ti.ndrange(sim.particle_count):
        sim.update_first_order_coefficient(p)

@ti.kernel
def k_calculate_kelvin_term():
    for p in ti.ndrange(sim.particle_count):
        sim.update_kelvin_term(p)

@ti.kernel
def k_calculate_pressure_delta():
    for p in ti.ndrange(sim.particle_count):
        sim.update_partial_pressure(p)

@ti.kernel
def k_calculate_mass_transport_rate():
    for p in ti.ndrange(sim.particle_count):
        sim.update_mass_transport_rate(p)

@ti.kernel
def k_calculate_scaling_factors():
    sim.update_scaling_factors(sim.time_step)

@ti.kernel
def k_calculate_transferable_mass():
    for p in ti.ndrange(sim.particle_count):
        sim.update_transferable_mass(p, sim.time_step)

@ti.kernel
def k_calculate_gas_mass():
    sim.update_gas_mass()

@ti.kernel
def k_calculate_species_masses():
    sim.update_species_masses()

@ti.kernel
def k_simulation_step():
    sim.simulation_step()

@ti.kernel
def k_fused_step():
    sim.fused_step()

class TestParticleResolvedKernels(unittest.TestCase):

    def test_radius_positive(self):
        k_calculate_radius()
        self.assertTrue((sim.radius.to_numpy() > 0.0).all())

    def test_first_order_coefficient(self):
        # radius is prerequisite
        k_calculate_radius()
        k_calculate_first_order_coefficient()
        coef = sim.first_order_coefficient.to_numpy()
        self.assertTrue(np.isfinite(coef).all())
        self.assertTrue((coef != 0.0).any())

    def test_kelvin_term(self):
        k_calculate_radius()
        k_calculate_kelvin_term()
        kel = sim.kelvin_term.to_numpy()
        self.assertTrue(np.isfinite(kel).all())

    def test_pressure_delta(self):
        k_calculate_radius()
        k_calculate_kelvin_term()
        k_calculate_pressure_delta()
        dP = sim.pressure_delta.to_numpy()
        self.assertTrue(np.isfinite(dP).all())

    def test_mass_transport_rate(self):
        k_calculate_radius()
        k_calculate_first_order_coefficient()
        k_calculate_kelvin_term()
        k_calculate_pressure_delta()
        k_calculate_mass_transport_rate()
        mtr = sim.mass_transport_rate.to_numpy()
        self.assertTrue(np.isfinite(mtr).all())

    def test_scaling_factors(self):
        k_calculate_radius()
        k_calculate_first_order_coefficient()
        k_calculate_kelvin_term()
        k_calculate_pressure_delta()
        k_calculate_mass_transport_rate()
        k_calculate_scaling_factors()
        sf = sim.scaling_factor.to_numpy()
        self.assertTrue(np.isfinite(sf).all())
        self.assertTrue(((sf >= 0.0) & (sf <= 1.0)).all())

    def test_transferable_mass(self):
        # full chain up to transferable mass
        k_calculate_radius()
        k_calculate_first_order_coefficient()
        k_calculate_kelvin_term()
        k_calculate_pressure_delta()
        k_calculate_mass_transport_rate()
        k_calculate_scaling_factors()
        k_calculate_transferable_mass()
        tm = sim.transferable_mass.to_numpy()
        self.assertTrue(np.isfinite(tm).all())
        self.assertTrue((tm != 0.0).any())

    def test_full_simulation_step(self):
        k_simulation_step()         # functional (unfused) path
        unfused_tm = sim.transferable_mass.to_numpy().copy()
        k_fused_step()              # fused path
        fused_tm = sim.transferable_mass.to_numpy()
        # verify both execution paths give identical results
        np.testing.assert_allclose(unfused_tm, fused_tm, rtol=1e-12, atol=1e-12)

if __name__ == "__main__":
    unittest.main()
