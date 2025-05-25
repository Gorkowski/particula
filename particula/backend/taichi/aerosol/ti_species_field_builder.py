
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
        """Fill variant `v` (row) with NumPy arrays (shape = [species])."""

        def _set_row(field_attr, values):
            full = field_attr.to_numpy()          # shape = (variants, species)
            full[v, :] = values                   # overwrite one row
            field_attr.from_numpy(full)

        _set_row(self.field.density,            density)
        _set_row(self.field.molar_mass,         molar_mass)
        _set_row(self.field.pure_vapor_pressure, pure_vapor_pressure)
        _set_row(self.field.vapor_concentration, vapor_concentration)
        _set_row(self.field.kappa,              kappa)
        _set_row(self.field.surface_tension,    surface_tension)
        _set_row(self.field.gas_mass,           gas_mass)
