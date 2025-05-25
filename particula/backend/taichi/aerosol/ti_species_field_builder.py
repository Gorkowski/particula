
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
        self.fields = [self.Species.field(shape=(species_count,))
                       for _ in range(variant_count)]

    @property
    def field(self):
        raise AttributeError(
            "SpeciesFieldBuilder now stores a list of fields; "
            "use `builder.fields[i]` or `builder.variant(i)`."
        )

    def load(self, v:int, *, density, molar_mass, pure_vapor_pressure,
             vapor_concentration, kappa, surface_tension, gas_mass):
        if not 0 <= v < self.variant_count:
            raise IndexError("variant index out of range")
        fld = self.fields[v]
        # upload (all inputs must be 1-D length = species_count)
        fld.density.from_numpy(np.ascontiguousarray(density, dtype=np.float32))
        fld.molar_mass.from_numpy(np.ascontiguousarray(molar_mass, dtype=np.float32))
        fld.pure_vapor_pressure.from_numpy(
            np.ascontiguousarray(pure_vapor_pressure, dtype=np.float32))
        fld.vapor_concentration.from_numpy(
            np.ascontiguousarray(vapor_concentration, dtype=np.float32))
        fld.kappa.from_numpy(np.ascontiguousarray(kappa, dtype=np.float32))
        fld.surface_tension.from_numpy(
            np.ascontiguousarray(surface_tension, dtype=np.float32))
        fld.gas_mass.from_numpy(np.ascontiguousarray(gas_mass, dtype=np.float32))

    def variant(self, v:int):
        return self.fields[v]
