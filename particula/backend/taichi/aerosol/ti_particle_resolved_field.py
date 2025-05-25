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
            species_masses=ti.f32,  # species_masses
            mass_transport_rate=ti.f32,  # mass_transport_rate
            transferable_mass=ti.f32,  # transferable_mass
        )
        self.field = self.P2S.field(
            shape=(variant_count, particle_count, species_count)
        )

        # ensure auxiliary sub-fields start at zero
        self.field.mass_transport_rate.fill(0.0)
        self.field.transferable_mass.fill(0.0)

    def load(self, v: int, *, species_masses: np.ndarray) -> None:
        """Copy `(particle, species)` mass matrix for one variant."""
        # validate input
        if species_masses.shape != (self.particle_count, self.species_count):
            raise ValueError(
                f"species_masses must have shape "
                f"({self.particle_count}, {self.species_count})"
            )

        # --- fast copy: overwrite whole variant row then push back ---
        full = self.field.species_masses.to_numpy()          # shape = (V, P, S)
        full[v, :, :] = species_masses.astype(np.float32)     # overwrite one 2-D slab
        self.field.species_masses.from_numpy(full)
