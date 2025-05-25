"""
Field for particle resolved aerosol simulation using Taichi.
"""

import taichi as ti
import numpy as np


class ParticleResolvedFieldBuilder:
    """Creates the (variant × particle × species) state field."""

    def __init__(
        self, variant_count: int, particle_count: int, species_count: int
    ):
        self.variant_count = variant_count
        self.particle_count = particle_count
        self.species_count = species_count

        self.P2S = ti.types.struct(
            mass=ti.f32,  # species_masses
            mtr=ti.f32,  # mass_transport_rate
            t_mass=ti.f32,  # transferable_mass
        )
        self.field = self.P2S.field(
            shape=(variant_count, particle_count, species_count)
        )

    def load(self, v: int, *, species_masses: np.ndarray) -> None:
        """Copy `(particle, species)` mass matrix for one variant."""
        self.field.mass[v].from_numpy(species_masses)
        self.field.mtr[v].fill(0.0)  # mass transport rate
        self.field.t_mass[v].fill(0.0)  # transferable mass
