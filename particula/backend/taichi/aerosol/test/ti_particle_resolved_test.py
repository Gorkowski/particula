import unittest
import taichi as ti
import numpy as np

# import the simulation module (already initialises fields/data)
from particula.backend.taichi.aerosol import ti_particle_resolved as sim

# build random inputs the same size as the module constants
P, S = sim.particle_count, sim.species_count
rand = np.random.rand
sim_obj = sim.TiAerosolParticleResolved(
    rand(P, S),
    rand(S),
    np.abs(rand(S)),
    np.abs(rand(S)),
    np.abs(rand(S)),
    np.abs(rand(S)),
    np.abs(rand(S)),
    np.abs(rand(S)),
    np.ones(P),
)

@ti.kernel
def k_calculate_radius(obj: ti.template()):
    for p in ti.ndrange(obj.species_masses.shape[0]):
        obj.update_radius(p)

@ti.kernel
def k_calculate_first_order_coefficient(obj: ti.template()):
    for p in ti.ndrange(obj.species_masses.shape[0]):
        obj.update_first_order_coefficient(p)

@ti.kernel
def k_calculate_kelvin_term(obj: ti.template()):
    for p in ti.ndrange(obj.species_masses.shape[0]):
        obj.update_kelvin_term(p)

@ti.kernel
def k_calculate_pressure_delta(obj: ti.template()):
    for p in ti.ndrange(obj.species_masses.shape[0]):
        obj.update_partial_pressure(p)

@ti.kernel
def k_calculate_mass_transport_rate(obj: ti.template()):
    for p in ti.ndrange(obj.species_masses.shape[0]):
        obj.update_mass_transport_rate(p)

@ti.kernel
def k_calculate_scaling_factors(obj: ti.template()):
    obj.update_scaling_factors(obj.time_step)

@ti.kernel
def k_calculate_transferable_mass(obj: ti.template()):
    for p in ti.ndrange(obj.species_masses.shape[0]):
        obj.update_transferable_mass(p, obj.time_step)

@ti.kernel
def k_calculate_gas_mass(obj: ti.template()):
    obj.update_gas_mass()

@ti.kernel
def k_calculate_species_masses(obj: ti.template()):
    obj.update_species_masses()

class TestParticleResolvedKernels(unittest.TestCase):

    def test_radius_positive(self):
        k_calculate_radius(sim_obj)
        self.assertTrue((sim_obj.radius.to_numpy() > 0.0).all())

    def test_first_order_coefficient(self):
        # radius is prerequisite
        k_calculate_radius(sim_obj)
        k_calculate_first_order_coefficient(sim_obj)
        coef = sim_obj.first_order_coefficient.to_numpy()
        self.assertTrue(np.isfinite(coef).all())
        self.assertTrue((coef != 0.0).any())

    def test_kelvin_term(self):
        k_calculate_radius(sim_obj)
        k_calculate_kelvin_term(sim_obj)
        kel = sim_obj.kelvin_term.to_numpy()
        self.assertTrue(np.isfinite(kel).all())

    def test_pressure_delta(self):
        k_calculate_radius(sim_obj)
        k_calculate_kelvin_term(sim_obj)
        k_calculate_pressure_delta(sim_obj)
        dP = sim_obj.pressure_delta.to_numpy()
        self.assertTrue(np.isfinite(dP).all())

    def test_mass_transport_rate(self):
        k_calculate_radius(sim_obj)
        k_calculate_first_order_coefficient(sim_obj)
        k_calculate_kelvin_term(sim_obj)
        k_calculate_pressure_delta(sim_obj)
        k_calculate_mass_transport_rate(sim_obj)
        mtr = sim_obj.mass_transport_rate.to_numpy()
        self.assertTrue(np.isfinite(mtr).all())

    def test_scaling_factors(self):
        k_calculate_radius(sim_obj)
        k_calculate_first_order_coefficient(sim_obj)
        k_calculate_kelvin_term(sim_obj)
        k_calculate_pressure_delta(sim_obj)
        k_calculate_mass_transport_rate(sim_obj)
        k_calculate_scaling_factors(sim_obj)
        sf = sim_obj.scaling_factor.to_numpy()
        self.assertTrue(np.isfinite(sf).all())
        self.assertTrue(((sf >= 0.0) & (sf <= 1.0)).all())

    def test_transferable_mass(self):
        # full chain up to transferable mass
        k_calculate_radius(sim_obj)
        k_calculate_first_order_coefficient(sim_obj)
        k_calculate_kelvin_term(sim_obj)
        k_calculate_pressure_delta(sim_obj)
        k_calculate_mass_transport_rate(sim_obj)
        k_calculate_scaling_factors(sim_obj)
        k_calculate_transferable_mass(sim_obj)
        tm = sim_obj.transferable_mass.to_numpy()
        self.assertTrue(np.isfinite(tm).all())
        self.assertTrue((tm != 0.0).any())

    def test_full_simulation_step(self):
        sim_obj.simulation_step()         # functional (unfused) path
        unfused_tm = sim_obj.transferable_mass.to_numpy().copy()
        sim_obj.fused_step()              # fused path
        fused_tm = sim_obj.transferable_mass.to_numpy()
        # verify both execution paths give identical results
        np.testing.assert_allclose(unfused_tm, fused_tm, rtol=1e-12, atol=1e-12)

if __name__ == "__main__":
    unittest.main()
