"""
High-level tests for TiAerosolParticleResolved_soa.

We verify that the builder-based `setup` succeeds and that the internal
Taichi fields contain the expected data for both single-variant and
multi-variant cases.  No physics is evaluated here – we only check data
placement and basic getters.
"""

import unittest
import numpy as np
import taichi as ti

from particula.backend.taichi.aerosol.ti_particle_resolved import (
    TiAerosolParticleResolved,
)

ti.init(arch=ti.cpu, default_fp=ti.f64)


class TestTiAerosolParticleResolvedSOA(unittest.TestCase):
    # ------------------------------------------------------------------
    # shared helpers & base arrays
    # ------------------------------------------------------------------
    def setUp(self) -> None:
        self.particles = 3
        self.species = 2

        # per-particle × species mass matrix
        self.base_mass = np.array(
            [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]], dtype=np.float32
        )

        # per-species 1-D arrays
        self.base_density = np.array([1.0, 1.1], dtype=np.float32)
        self.base_molar_mass = np.array([2.0, 2.1], dtype=np.float32)
        self.base_pvp = np.array([3.0, 3.1], dtype=np.float32)
        self.base_vapor_conc = np.array([4.0, 4.1], dtype=np.float32)
        self.base_kappa = np.array([0.1, 0.2], dtype=np.float32)
        self.base_sigma = np.array([6.0, 6.1], dtype=np.float32)
        self.base_gas_mass = np.array([7.0, 7.1], dtype=np.float32)

        # per-particle concentration (1-D)
        self.base_particle_conc = np.array(
            [10.0, 11.0, 12.0], dtype=np.float32
        )

    # ------------------------------------------------------------------
    # internal helper: build & fill a solver with a chosen #variants
    # ------------------------------------------------------------------
    def _make_solver(self, variants: int) -> TiAerosolParticleResolved:
        sim = TiAerosolParticleResolved(
            particle_count=self.particles,
            species_count=self.species,
            variant_count=variants,
        )

        # load each variant with an easily recognisable offset
        for v in range(variants):
            off = float(v) * 10.0
            sim.setup(
                variant_index=v,
                species_masses_np=self.base_mass + off,
                density_np=self.base_density + off,
                molar_mass_np=self.base_molar_mass + off,
                pure_vapor_pressure_np=self.base_pvp + off,
                vapor_concentration_np=self.base_vapor_conc + off,
                kappa_value_np=self.base_kappa,
                surface_tension_np=self.base_sigma + off,
                gas_mass_np=self.base_gas_mass + off,
                particle_concentration_np=self.base_particle_conc + off,
            )
        return sim

    # ------------------------------------------------------------------
    # single-variant checks
    # ------------------------------------------------------------------
    def test_single_variant_setup(self):
        sim = self._make_solver(variants=1)

        # shapes
        self.assertEqual(len(sim.species), 1)
        self.assertEqual(sim.species[0].shape, (self.species,))

        self.assertEqual(len(sim.particle_field), 1)
        self.assertEqual(
            sim.particle_field[0].species_masses.shape,
            (self.particles, self.species),
        )

        # data correctness
        np.testing.assert_array_equal(
            sim.species[0].density.to_numpy(), self.base_density
        )
        np.testing.assert_array_equal(
            sim.particle_field[0].species_masses.to_numpy(), self.base_mass
        )

    # ------------------------------------------------------------------
    # multi-variant checks
    # ------------------------------------------------------------------
    def test_multiple_variant_setup(self):
        n_variants = 4
        sim = self._make_solver(variants=n_variants)

        for v in range(n_variants):
            off = float(v) * 10.0
            np.testing.assert_array_equal(
                sim.species[v].density.to_numpy(), self.base_density + off
            )
            np.testing.assert_array_equal(
                sim.particle_field[v].species_masses.to_numpy(),
                self.base_mass + off,
            )

    # ------------------------------------------------------------------
    # fused-step checks
    # ------------------------------------------------------------------
    def test_single_variant_fused_step(self):
        """`fused_step` should compile & run for a single variant."""
        sim = self._make_solver(variants=1)
        sim.fused_step()  # run once
        assert True

    def test_multiple_variant_fused_step(self):
        """`fused_step` should compile & run for several variants."""
        n_variants = 4
        sim = self._make_solver(variants=n_variants)
        sim.fused_step()
        for v in range(n_variants):
            r = sim.get_radius(v)
            self.assertEqual(r.shape, (self.particles,))
            self.assertTrue(np.all(r > 0.0))


if __name__ == "__main__":
    unittest.main()
