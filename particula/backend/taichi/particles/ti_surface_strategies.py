"""Taichi drop-in for particula.particles.surface_strategies."""

import taichi as ti
import numpy as np
from typing import Union
from numpy.typing import NDArray
from particula.backend.dispatch_register import register

if not ti.core.is_initialized():
    ti.init(arch=ti.cpu, default_fp=ti.f64)      # initialise once, float-64 everywhere

from particula.backend.taichi.particles.properties.ti_kelvin_effect_module import (
    fget_kelvin_radius,
    fget_kelvin_term,
)


def _store(value):
    if np.isscalar(value):
        fld = ti.field(ti.f64, shape=())
        fld[None] = float(value)
        return fld, 1
    arr = np.asarray(value, dtype=np.float64, order="C")
    fld = ti.field(ti.f64, shape=arr.size)
    for i in range(arr.size):
        fld[i] = arr[i]
    return fld, arr.size


@ti.data_oriented
class _SurfaceMixin:
    """Shared helpers reused by the three pure-surface strategies."""

    def __init__(self, surface_tension, density):
        self.surface_tension, self.n_species = _store(surface_tension)
        self.density, _ = _store(density)

    @ti.func
    def _kelvin_radius_elem(
        self, sigma: ti.f64, rho: ti.f64, M: ti.f64, T: ti.f64
    ) -> ti.f64:
        # Delegates to the validated helper – keeps one single source of truth
        return fget_kelvin_radius(sigma, rho, M, T)

    @ti.func
    def _kelvin_term_elem(self, r_p: ti.f64, r_k: ti.f64) -> ti.f64:
        return fget_kelvin_term(r_p, r_k)

    def get_name(self) -> str:
        return self.__class__.__name__


@register("SurfaceStrategyMolar", backend="taichi")
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
        mc = np.asarray(mass_concentration, dtype=np.float64)
        mm = self.molar_mass.to_numpy()
        x = mc / mm
        x /= x.sum()
        return (self.surface_tension.to_numpy() * x).sum(dtype=np.float64)

    def effective_density(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> float:
        if self.n_species == 1:
            return self.density[None]
        mc = np.asarray(mass_concentration, dtype=np.float64)
        mm = self.molar_mass.to_numpy()
        x = mc / mm
        x /= x.sum()
        return (self.density.to_numpy() * x).sum(dtype=np.float64)

    def kelvin_radius(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        mass_concentration: Union[float, NDArray[np.float64]],
        temperature: float,
    ) -> Union[float, NDArray[np.float64]]:
        sigma = self.effective_surface_tension(mass_concentration)
        rho = self.effective_density(mass_concentration)
        return fget_kelvin_radius(
            sigma, rho, molar_mass, temperature
        )

    def kelvin_term(
        self,
        radius: Union[float, NDArray[np.float64]],
        molar_mass: Union[float, NDArray[np.float64]],
        mass_concentration: Union[float, NDArray[np.float64]],
        temperature: float,
    ) -> Union[float, NDArray[np.float64]]:
        r_k = self.kelvin_radius(molar_mass, mass_concentration, temperature)
        return np.exp(
            fget_kelvin_term(radius, r_k)          # preserves NumPy broadcasting
        )


@register("SurfaceStrategyMass", backend="taichi")
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
        mc = np.asarray(mass_concentration, dtype=np.float64)
        x = mc / mc.sum()
        return (self.surface_tension.to_numpy() * x).sum(dtype=np.float64)

    def effective_density(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> float:
        if self.n_species == 1:
            return self.density[None]
        mc = np.asarray(mass_concentration, dtype=np.float64)
        x = mc / mc.sum()
        return (self.density.to_numpy() * x).sum(dtype=np.float64)

    def kelvin_radius(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        mass_concentration: Union[float, NDArray[np.float64]],
        temperature: float,
    ) -> Union[float, NDArray[np.float64]]:
        sigma = self.effective_surface_tension(mass_concentration)
        rho = self.effective_density(mass_concentration)
        return fget_kelvin_radius(
            sigma, rho, molar_mass, temperature
        )

    def kelvin_term(
        self,
        radius: Union[float, NDArray[np.float64]],
        molar_mass: Union[float, NDArray[np.float64]],
        mass_concentration: Union[float, NDArray[np.float64]],
        temperature: float,
    ) -> Union[float, NDArray[np.float64]]:
        r_k = self.kelvin_radius(molar_mass, mass_concentration, temperature)
        return np.exp(
            fget_kelvin_term(radius, r_k)          # preserves NumPy broadcasting
        )


@register("SurfaceStrategyVolume", backend="taichi")
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
        mc = np.asarray(mass_concentration, dtype=np.float64)
        dens = self.density.to_numpy()
        vol = mc / dens
        x = vol / vol.sum()
        return (self.surface_tension.to_numpy() * x).sum(dtype=np.float64)

    def effective_density(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> float:
        if self.n_species == 1:
            return self.density[None]
        mc = np.asarray(mass_concentration, dtype=np.float64)
        dens = self.density.to_numpy()
        vol = mc / dens
        x = vol / vol.sum()
        return (self.density.to_numpy() * x).sum(dtype=np.float64)

    def kelvin_radius(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        mass_concentration: Union[float, NDArray[np.float64]],
        temperature: float,
    ) -> Union[float, NDArray[np.float64]]:
        sigma = self.effective_surface_tension(mass_concentration)
        rho = self.effective_density(mass_concentration)
        return fget_kelvin_radius(
            sigma, rho, molar_mass, temperature
        )

    def kelvin_term(
        self,
        radius: Union[float, NDArray[np.float64]],
        molar_mass: Union[float, NDArray[np.float64]],
        mass_concentration: Union[float, NDArray[np.float64]],
        temperature: float,
    ) -> Union[float, NDArray[np.float64]]:
        r_k = self.kelvin_radius(molar_mass, mass_concentration, temperature)
        return np.exp(
            fget_kelvin_term(radius, r_k)          # preserves NumPy broadcasting
        )
