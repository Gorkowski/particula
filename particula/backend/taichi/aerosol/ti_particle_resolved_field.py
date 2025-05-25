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

        P2S = ti.types.struct(
            species_masses=ti.f32,  # species_masses
            mass_transport_rate=ti.f32,  # mass_transport_rate
            transferable_mass=ti.f32,  # transferable_mass
        )
        self.fields = [
            P2S.field(shape=(particle_count, species_count))
            for _ in range(variant_count)
        ]

        # ensure auxiliary sub-fields start at zero
        for fld in self.fields:
            fld.mass_transport_rate.fill(0.0)
            fld.transferable_mass.fill(0.0)

    def load(self, v: int, *, species_masses: np.ndarray) -> None:
        """Copy `(particle, species)` mass matrix for one variant."""
        if species_masses.shape != (self.particle_count, self.species_count):
            raise ValueError("species_masses must have shape "
                             f"({self.particle_count}, {self.species_count})")
        self.fields[v].species_masses.from_numpy(
            np.ascontiguousarray(species_masses, dtype=np.float32)
        )
    def variant(self, v: int):
        return self.fields[v]
