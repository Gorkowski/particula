
"""Aerosol species field builder for Taichi backend."""

import taichi as ti
import numpy as np



class SpeciesFieldBuilder:
    """Creates the per-species SoA field and offers handy load helpers."""

    def __init__(self, variant_count: int, species_count: int):
        self.variant_count = variant_count
        self.species_count = species_count

        self.Species = ti.types.struct(
            density=ti.f32,
            molar_mass=ti.f32,
            pure_vapor_pressure=ti.f32,
            vapor_concentration=ti.f32,
            kappa=ti.f32,
            surface_tension=ti.f32,
            gas_mass=ti.f32,
        )
        self.field = self.Species.field(shape=(variant_count, species_count))

    # ------------ helper: copy one variant from NumPy -----------------
    def load(
        self,
        v: int,
        *,
        density: np.ndarray,
        molar_mass: np.ndarray,
        pure_vapor_pressure: np.ndarray,
        vapor_concentration: np.ndarray,
        kappa: np.ndarray,
        surface_tension: np.ndarray,
        gas_mass: np.ndarray,
    ) -> None:
        """Fill variant `v` with NumPy arrays (1-D, length = species)."""
        self.field.density[v].from_numpy(density)
        self.field.molar_mass[v].from_numpy(molar_mass)
        self.field.pure_vapor_pressure[v].from_numpy(pure_vapor_pressure)
        self.field.vapor_concentration[v].from_numpy(vapor_concentration)
        self.field.kappa[v].from_numpy(kappa)
        self.field.surface_tension[v].from_numpy(surface_tension)
        self.field.gas_mass[v].from_numpy(gas_mass)
