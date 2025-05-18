"""Taichi drop-in for particula.particles.surface_strategies."""

import taichi as ti
import numpy as np
from typing import Union
from numpy.typing import NDArray

from particula.backend.taichi.particles.properties.ti_kelvin_effect_module import (
    ti_get_kelvin_radius,          # <- NEW
    ti_get_kelvin_term,            # <- NEW
    fget_kelvin_radius,            # keep: still used in in-kernel helpers
    fget_kelvin_term,
)


def _store(value):
    if np.isscalar(value):
        ti_field = ti.field(ti.f64, shape=())
        ti_field[None] = float(value)
        return ti_field, 1
    np_array = np.asarray(value, dtype=np.float64, order="C")
    ti_field = ti.field(ti.f64, shape=np_array.size)
    for i in range(np_array.size):
        ti_field[i] = np_array[i]
    return ti_field, np_array.size


@ti.data_oriented
class _SurfaceMixin:
    """Shared helpers reused by the three pure-surface strategies."""

    def __init__(self, surface_tension, density):
        self.surface_tension, self.n_species = _store(surface_tension)
        self.density, _ = _store(density)

    @ti.func
    def _kelvin_radius_elem(
        self,
        surface_tension: ti.f64,
        density: ti.f64,
        molar_mass: ti.f64,
        temperature: ti.f64,
    ) -> ti.f64:
        # Delegates to the validated helper – keeps one single source of truth
        return fget_kelvin_radius(
            surface_tension, density, molar_mass, temperature
        )

    @ti.func
    def _kelvin_term_elem(
        self,
        particle_radius: ti.f64,
        kelvin_radius_value: ti.f64,
    ) -> ti.f64:
        return fget_kelvin_term(particle_radius, kelvin_radius_value)

    def get_name(self) -> str:
        return self.__class__.__name__


@ti.data_oriented
class SurfaceStrategyMolar(_SurfaceMixin):
    def __init__(
        self,
        surface_tension: Union[float, NDArray[np.float64]] = 0.072,
        density: Union[float, NDArray[np.float64]] = 1000,
        molar_mass: Union[float, NDArray[np.float64]] = 0.01815,
    ):
        super().__init__(surface_tension, density)
        self.molar_mass, _ = _store(molar_mass)

    def effective_surface_tension(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> float:
        if self.n_species == 1:
            return self.surface_tension[None]
        mass_concentration_array = np.asarray(mass_concentration, dtype=np.float64)
        molar_mass_array = self.molar_mass.to_numpy()
        mole_fraction = mass_concentration_array / molar_mass_array
        mole_fraction /= mole_fraction.sum()
        return (self.surface_tension.to_numpy() * mole_fraction).sum(dtype=np.float64)

    def effective_density(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> float:
        if self.n_species == 1:
            return self.density[None]
        mass_concentration_array = np.asarray(mass_concentration, dtype=np.float64)
        molar_mass_array = self.molar_mass.to_numpy()
        mole_fraction = mass_concentration_array / molar_mass_array
        mole_fraction /= mole_fraction.sum()
        return (self.density.to_numpy() * mole_fraction).sum(dtype=np.float64)

    def kelvin_radius(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        mass_concentration: Union[float, NDArray[np.float64]],
        temperature: float,
    ) -> Union[float, NDArray[np.float64]]:
        surface_tension_value = self.effective_surface_tension(mass_concentration)
        effective_density_value = self.effective_density(mass_concentration)
        # convert to 1-D float64 arrays –  ti_get_kelvin_radius needs ndarrays
        surface_tension_array = np.atleast_1d(surface_tension_value).astype(np.float64)
        density_array = np.atleast_1d(effective_density_value).astype(np.float64)
        molar_mass_array = np.atleast_1d(molar_mass).astype(np.float64)
        return ti_get_kelvin_radius(
            surface_tension_array, density_array, molar_mass_array, float(temperature)
        )

    def kelvin_term(
        self,
        radius: Union[float, NDArray[np.float64]],
        molar_mass: Union[float, NDArray[np.float64]],
        mass_concentration: Union[float, NDArray[np.float64]],
        temperature: float,
    ) -> Union[float, NDArray[np.float64]]:
        kelvin_radius_value = self.kelvin_radius(
            molar_mass, mass_concentration, temperature
        )
        particle_radius_array = np.atleast_1d(radius).astype(np.float64)
        kelvin_radius_array = np.atleast_1d(kelvin_radius_value).astype(np.float64)
        return ti_get_kelvin_term(particle_radius_array, kelvin_radius_array)


@ti.data_oriented
class SurfaceStrategyMass(_SurfaceMixin):
    def __init__(
        self,
        surface_tension: Union[float, NDArray[np.float64]] = 0.072,
        density: Union[float, NDArray[np.float64]] = 1000,
    ):
        super().__init__(surface_tension, density)

    def effective_surface_tension(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> float:
        if self.n_species == 1:
            return self.surface_tension[None]
        mass_concentration_array = np.asarray(mass_concentration, dtype=np.float64)
        mass_fraction = mass_concentration_array / mass_concentration_array.sum()
        return (self.surface_tension.to_numpy() * mass_fraction).sum(dtype=np.float64)

    def effective_density(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> float:
        if self.n_species == 1:
            return self.density[None]
        mass_concentration_array = np.asarray(mass_concentration, dtype=np.float64)
        mass_fraction = mass_concentration_array / mass_concentration_array.sum()
        return (self.density.to_numpy() * mass_fraction).sum(dtype=np.float64)

    def kelvin_radius(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        mass_concentration: Union[float, NDArray[np.float64]],
        temperature: float,
    ) -> Union[float, NDArray[np.float64]]:
        surface_tension_value = self.effective_surface_tension(mass_concentration)
        effective_density_value = self.effective_density(mass_concentration)
        # convert to 1-D float64 arrays –  ti_get_kelvin_radius needs ndarrays
        surface_tension_array = np.atleast_1d(surface_tension_value).astype(np.float64)
        density_array = np.atleast_1d(effective_density_value).astype(np.float64)
        molar_mass_array = np.atleast_1d(molar_mass).astype(np.float64)
        return ti_get_kelvin_radius(
            surface_tension_array, density_array, molar_mass_array, float(temperature)
        )

    def kelvin_term(
        self,
        radius: Union[float, NDArray[np.float64]],
        molar_mass: Union[float, NDArray[np.float64]],
        mass_concentration: Union[float, NDArray[np.float64]],
        temperature: float,
    ) -> Union[float, NDArray[np.float64]]:
        kelvin_radius_value = self.kelvin_radius(
            molar_mass, mass_concentration, temperature
        )
        particle_radius_array = np.atleast_1d(radius).astype(np.float64)
        kelvin_radius_array = np.atleast_1d(kelvin_radius_value).astype(np.float64)
        return ti_get_kelvin_term(particle_radius_array, kelvin_radius_array)


@ti.data_oriented
class SurfaceStrategyVolume(_SurfaceMixin):
    def __init__(
        self,
        surface_tension: Union[float, NDArray[np.float64]] = 0.072,
        density: Union[float, NDArray[np.float64]] = 1000,
    ):
        super().__init__(surface_tension, density)

    def effective_surface_tension(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> float:
        if self.n_species == 1:
            return self.surface_tension[None]
        mass_concentration_array = np.asarray(mass_concentration, dtype=np.float64)
        density_array = self.density.to_numpy()
        volume_array = mass_concentration_array / density_array
        volume_fraction_weight = volume_array / volume_array.sum()
        return (self.surface_tension.to_numpy() * volume_fraction_weight).sum(dtype=np.float64)

    def effective_density(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> float:
        if self.n_species == 1:
            return self.density[None]
        mass_concentration_array = np.asarray(mass_concentration, dtype=np.float64)
        density_array = self.density.to_numpy()
        volume_array = mass_concentration_array / density_array
        volume_fraction_weight = volume_array / volume_array.sum()
        return (self.density.to_numpy() * volume_fraction_weight).sum(dtype=np.float64)

    def kelvin_radius(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        mass_concentration: Union[float, NDArray[np.float64]],
        temperature: float,
    ) -> Union[float, NDArray[np.float64]]:
        surface_tension_value = self.effective_surface_tension(mass_concentration)
        effective_density_value = self.effective_density(mass_concentration)
        # convert to 1-D float64 arrays –  ti_get_kelvin_radius needs ndarrays
        surface_tension_array = np.atleast_1d(surface_tension_value).astype(np.float64)
        density_array = np.atleast_1d(effective_density_value).astype(np.float64)
        molar_mass_array = np.atleast_1d(molar_mass).astype(np.float64)
        return ti_get_kelvin_radius(
            surface_tension_array, density_array, molar_mass_array, float(temperature)
        )

    def kelvin_term(
        self,
        radius: Union[float, NDArray[np.float64]],
        molar_mass: Union[float, NDArray[np.float64]],
        mass_concentration: Union[float, NDArray[np.float64]],
        temperature: float,
    ) -> Union[float, NDArray[np.float64]]:
        kelvin_radius_value = self.kelvin_radius(
            molar_mass, mass_concentration, temperature
        )
        particle_radius_array = np.atleast_1d(radius).astype(np.float64)
        kelvin_radius_array = np.atleast_1d(kelvin_radius_value).astype(np.float64)
        return ti_get_kelvin_term(particle_radius_array, kelvin_radius_array)
