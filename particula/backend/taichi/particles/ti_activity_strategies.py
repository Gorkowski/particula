"""Taichi implementation of particula.particles.activity_strategies."""

import taichi as ti
import numpy as np
from typing import Union
from numpy.typing import NDArray
from particula.backend.dispatch_register import register

from particula.backend.taichi.particles.properties.ti_activity_module import (
    fget_surface_partial_pressure,
    kget_surface_partial_pressure,
)

ti.init(default_fp=ti.f64)          # enforce float64 everywhere


# ────────────────────────── shared mixin (no public ctor) ───────────────────
@ti.data_oriented
class _ActivityMixin:
    """Helpers common to every activity strategy."""

    # one-off element-wise helpers ------------------------------------------------
    @ti.func
    def _get_surface_partial_pressure(self, pure_vapor_pressure: ti.f64, activity: ti.f64) -> ti.f64:
        return fget_surface_partial_pressure(pure_vapor_pressure, activity)

    # vectorised kernel (1-D ndarray in / out) -----------------------------------
    @ti.kernel
    def _partial_pressure_kernel(
        self,
        pure_vapor_pressure: ti.types.ndarray(dtype=ti.f64, ndim=1),
        activity:           ti.types.ndarray(dtype=ti.f64, ndim=1),
        result:             ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        """Vectorised Raoult/Margules: p = a · p⁰ (element-wise)."""
        for i in ti.ndrange(result.shape[0]):
            result[i] = fget_surface_partial_pressure(
                pure_vapor_pressure[i], activity[i]
            )

    # public wrapper identical to NumPy API --------------------------------------
    def partial_pressure(
        self,
        pure_vapor_pressure: Union[float, NDArray[np.float64]],
        mass_concentration: Union[float, NDArray[np.float64]],
    ) -> Union[float, NDArray[np.float64]]:
        """Return surface vapour pressure given p⁰ and concentration."""
        if np.ndim(pure_vapor_pressure) == 0:                 # scalar case
            # do the computation directly in Python to stay outside Taichi
            return float(pure_vapor_pressure) * float(
                self._activity_func(float(mass_concentration))
            )

        # vector case – reuse activity kernel then bulk multiply
        activity = self.activity(mass_concentration)          # ndarray (same shape)
        result   = np.empty_like(pure_vapor_pressure, dtype=np.float64)
        self._partial_pressure_kernel(pure_vapor_pressure, activity, result)
        return result


# ───────────────────────────── concrete strategies ──────────────────────────
@ti.data_oriented
class ActivityIdealMolar(_ActivityMixin):
    """Taichi drop-in for ActivityIdealMolar."""

    def __init__(self, molar_mass: float | NDArray[np.float64] = 0.0):
        self.molar_mass = ti.field(ti.f64, shape=())
        self.molar_mass[None] = float(np.asarray(molar_mass))

    def _activity_func(self, mass_concentration: float) -> float:
        return 1.0

    @ti.kernel
    def _activity_kernel(
        self,
        mass_concentration: ti.types.ndarray(dtype=ti.f64, ndim=1),
        result: ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        molar_mass = self.molar_mass[None]
        total_moles = ti.f64(0.0)
        for i in range(mass_concentration.shape[0]):
            total_moles += mass_concentration[i] / molar_mass
        for i in range(mass_concentration.shape[0]):
            result[i] = 0.0 if total_moles == 0.0 else (mass_concentration[i] / molar_mass) / total_moles

    def activity(self, mass_concentration):
        if np.ndim(mass_concentration) == 0:
            return self._activity_func(float(mass_concentration))

        result = np.empty_like(mass_concentration, dtype=np.float64)
        self._activity_kernel(mass_concentration, result)
        return result


@ti.data_oriented
class ActivityIdealMass(_ActivityMixin):
    """Taichi drop-in for ActivityIdealMass (parameter-free)."""

    def _activity_func(self, mass_concentration: float) -> float:
        return 1.0

    @ti.kernel
    def _activity_kernel(
        self,
        mass_concentration: ti.types.ndarray(dtype=ti.f64, ndim=1),
        result: ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        total_mass = ti.f64(0.0)
        for i in range(mass_concentration.shape[0]):
            total_mass += mass_concentration[i]
        for i in range(mass_concentration.shape[0]):
            result[i] = 0.0 if total_mass == 0.0 else mass_concentration[i] / total_mass

    def activity(self, mass_concentration):
        if np.ndim(mass_concentration) == 0:
            return self._activity_func(float(mass_concentration))
        result = np.empty_like(mass_concentration, dtype=np.float64)
        self._activity_kernel(mass_concentration, result)
        return result


@ti.data_oriented
class ActivityIdealVolume(_ActivityMixin):
    """Taichi drop-in for ActivityIdealVolume."""

    def __init__(self, density: float | NDArray[np.float64] = 0.0):
        self.density = ti.field(ti.f64, shape=())
        self.density[None] = float(np.asarray(density))

    def _activity_func(self, mass_concentration: float) -> float:
        return 1.0

    @ti.kernel
    def _activity_kernel(
        self,
        mass_concentration: ti.types.ndarray(dtype=ti.f64, ndim=1),
        result: ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        density = self.density[None]
        total_volume = ti.f64(0.0)
        for i in range(mass_concentration.shape[0]):
            total_volume += mass_concentration[i] / density
        for i in range(mass_concentration.shape[0]):
            result[i] = 0.0 if total_volume == 0.0 else (mass_concentration[i] / density) / total_volume

    def activity(self, mass_concentration):
        if np.ndim(mass_concentration) == 0:
            return self._activity_func(float(mass_concentration))
        result = np.empty_like(mass_concentration, dtype=np.float64)
        self._activity_kernel(mass_concentration, result)
        return result


@ti.data_oriented
class ActivityKappaParameter(_ActivityMixin):
    """Taichi drop-in for ActivityKappaParameter (non-ideal)."""

    def __init__(
        self,
        kappa:       NDArray[np.float64],
        density:     NDArray[np.float64],
        molar_mass:  NDArray[np.float64],
        water_index: int = 0,
    ):
        kappa       = np.asarray(kappa, dtype=np.float64)
        density     = np.asarray(density, dtype=np.float64)
        molar_mass  = np.asarray(molar_mass, dtype=np.float64)

        n = kappa.size
        self.kappa          = ti.field(ti.f64, shape=n)
        self.density        = ti.field(ti.f64, shape=n)
        self.molar_mass     = ti.field(ti.f64, shape=n)
        for i in range(n):
            self.kappa[i]      = kappa[i]
            self.density[i]    = density[i]
            self.molar_mass[i] = molar_mass[i]

        self.water_index = ti.field(dtype=ti.i32, shape=())
        self.water_index[None] = int(water_index)

    def _activity_func(self, mass_concentration: float) -> float:
        return 1.0

    @ti.kernel
    def _activity_kernel(
        self,
        mass_concentration: ti.types.ndarray(dtype=ti.f64, ndim=1),
        result: ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        n_species = mass_concentration.shape[0]
        water_index = self.water_index[None]

        # mole fractions -------------------------------------------------
        mol_sum = ti.f64(0.0)
        for s in range(n_species):
            mol_sum += mass_concentration[s] / self.molar_mass[s]
        for s in range(n_species):
            mol = mass_concentration[s] / self.molar_mass[s]
            result[s] = 0.0 if mol_sum == 0.0 else mol / mol_sum

        # κ-Köhler water activity ---------------------------------------
        volume_sum = ti.f64(0.0)
        for s in range(n_species):
            volume_sum += mass_concentration[s] / self.density[s]

        water_volume_fraction = ti.f64(0.0)
        if volume_sum > 0.0:
            water_volume_fraction = (mass_concentration[water_index] / self.density[water_index]) / volume_sum
        solute_volume_fraction = 1.0 - water_volume_fraction

        kappa_mixed = ti.f64(0.0)
        if solute_volume_fraction > 0.0:
            if n_species == 2:
                kappa_mixed = self.kappa[1 - water_index]
            else:
                for s in range(n_species):
                    if s != water_index:
                        volume_fraction_species = (mass_concentration[s] / self.density[s]) / volume_sum
                        kappa_mixed += (volume_fraction_species / solute_volume_fraction) * self.kappa[s]

        volume_term = 0.0
        if water_volume_fraction > 0.0:
            volume_term = kappa_mixed * solute_volume_fraction / water_volume_fraction

        result[water_index] = 0.0 if water_volume_fraction == 0.0 else 1.0 / (1.0 + volume_term)

    def activity(self, mass_concentration):
        if np.ndim(mass_concentration) == 0:
            return self._activity_func(float(mass_concentration))
        result = np.empty_like(mass_concentration, dtype=np.float64)
        self._activity_kernel(mass_concentration, result)
        return result


# ─── factory hooks for dispatch_register (one-liner each) ───────────────────
@register("ActivityIdealMolar", backend="taichi")
def _make_molar(*a, **k):      return ActivityIdealMolar(*a, **k)

@register("ActivityIdealMass", backend="taichi")
def _make_mass(*a, **k):       return ActivityIdealMass(*a, **k)

@register("ActivityIdealVolume", backend="taichi")
def _make_volume(*a, **k):        return ActivityIdealVolume(*a, **k)

@register("ActivityKappaParameter", backend="taichi")
def _make_kappa(*a, **k):      return ActivityKappaParameter(*a, **k)
