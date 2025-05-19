import unittest
import taichi as ti
import numpy as np
import numpy.testing as npt       # NEW

# import the simulation module (already initialises fields/data)
from particula.backend.taichi.aerosol import ti_particle_resolved as sim

# ----- random demo inputs (same sizes as constants in the module) -----
P, S = 20_000, 10
rand = np.random.rand
species_masses_np       = rand(P, S)
density_np              = rand(S)
molar_mass_np           = np.abs(rand(S))
pure_vapor_pressure_np  = np.abs(rand(S))
vapor_concentration_np  = np.abs(rand(S))
kappa_value_np          = np.abs(rand(S))
surface_tension_np      = np.abs(rand(S))
gas_mass_np             = np.abs(rand(S))
particle_concentration_np = np.ones(P)

# build the simulation object & populate fields
sim_obj = sim.TiAerosolParticleResolved(P, S)
sim_obj.setup(
    species_masses_np,
    density_np,
    molar_mass_np,
    pure_vapor_pressure_np,
    vapor_concentration_np,
    kappa_value_np,
    surface_tension_np,
    gas_mass_np,
    particle_concentration_np,
)

@ti.kernel
def k_calculate_radius(obj: ti.template()):
    n_p = ti.static(obj.species_masses.shape[0])
    for p in range(n_p):
        obj.update_radius(p)

@ti.kernel
def k_calculate_first_order_coefficient(obj: ti.template()):
    n_p = ti.static(obj.species_masses.shape[0])
    for p in range(n_p):
        obj.update_first_order_coefficient(p)

@ti.kernel
def k_calculate_kelvin_term(obj: ti.template()):
    n_p = ti.static(obj.species_masses.shape[0])
    for p in range(n_p):
        obj.update_kelvin_term(p)

@ti.kernel
def k_calculate_pressure_delta(obj: ti.template()):
    n_p = ti.static(obj.species_masses.shape[0])
    for p in range(n_p):
        obj.update_partial_pressure(p)

@ti.kernel
def k_calculate_mass_transport_rate(obj: ti.template()):
    n_p = ti.static(obj.species_masses.shape[0])
    for p in range(n_p):
        obj.update_mass_transport_rate(p)

@ti.kernel
def k_calculate_scaling_factors(obj: ti.template()):
    obj.update_scaling_factors(obj.time_step)

@ti.kernel
def k_calculate_transferable_mass(obj: ti.template()):
    n_p = ti.static(obj.species_masses.shape[0])
    for p in range(n_p):
        obj.update_transferable_mass(p, obj.time_step)

@ti.kernel
def k_calculate_gas_mass(obj: ti.template()):
    obj.update_gas_mass()

@ti.kernel
def k_calculate_species_masses(obj: ti.template()):
    obj.update_species_masses()

@ti.kernel
def k_fused_step(obj: ti.template()):
    obj.fused_step()

class TestParticleResolvedKernels(unittest.TestCase):

    def test_radius(self):
        k_calculate_radius(sim_obj)
        self.assertTrue((sim_obj.radius.to_numpy() > 0.0).all())

    def test_first_order_coefficient(self):
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
        dp = sim_obj.pressure_delta.to_numpy()
        self.assertTrue(np.isfinite(dp).all())

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

    def test_fused_vs_unfused(self):
        # Unfused path
        k_calculate_radius(sim_obj)
        k_calculate_first_order_coefficient(sim_obj)
        k_calculate_kelvin_term(sim_obj)
        k_calculate_pressure_delta(sim_obj)
        k_calculate_mass_transport_rate(sim_obj)
        k_calculate_scaling_factors(sim_obj)
        k_calculate_transferable_mass(sim_obj)
        k_calculate_gas_mass(sim_obj)
        k_calculate_species_masses(sim_obj)
        unfused_tm = sim_obj.transferable_mass.to_numpy().copy()

        # Fused path
        sim_obj.fused_step()
        fused_tm = sim_obj.transferable_mass.to_numpy()

        npt.assert_allclose(unfused_tm, fused_tm, rtol=1e-12, atol=1e-12)

if __name__ == "__main__":
    unittest.main()
