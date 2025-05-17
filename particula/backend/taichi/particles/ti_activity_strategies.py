"""Taichi implementation of particula.particles.activity_strategies."""

import taichi as ti
import numpy as np
from typing import Union
from numpy.typing import NDArray
from particula.backend.dispatch_register import register

# ─── element-wise & vector helpers already coded in ti_activity_module ──────
from particula.backend.taichi.particles.properties.ti_activity_module import (
    fget_ideal_activity_molar,
    kget_ideal_activity_molar,
    fget_ideal_activity_mass,
    kget_ideal_activity_mass,
    fget_ideal_activity_volume,
    kget_ideal_activity_volume,
    fget_kappa_activity,
    kget_kappa_activity,
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
    def _p_surface_func(self, p_pure: ti.f64, activity: ti.f64) -> ti.f64:
        return fget_surface_partial_pressure(p_pure, activity)

    # vectorised kernel (1-D ndarray in / out) -----------------------------------
    @ti.kernel
    def _partial_pressure_kernel(
        self,
        pure_vapor_pressure: ti.types.ndarray(dtype=ti.f64, ndim=1),
        activity:           ti.types.ndarray(dtype=ti.f64, ndim=1),
        result:             ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        """Vectorised Raoult/Margules: p = a · p⁰ (element-wise)."""
        kget_surface_partial_pressure(pure_vapor_pressure, activity, result)

    # public wrapper identical to NumPy API --------------------------------------
    def partial_pressure(
        self,
        pure_vapor_pressure: Union[float, NDArray[np.float64]],
        mass_concentration: Union[float, NDArray[np.float64]],
    ) -> Union[float, NDArray[np.float64]]:
        """Return surface vapour pressure given p⁰ and concentration."""
        if np.ndim(pure_vapor_pressure) == 0:                 # scalar case
            return self._p_surface_func(
                float(pure_vapor_pressure),
                self._activity_func(float(mass_concentration)),
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

    # element-wise
    @ti.func
    def _activity_func(self, mass_conc: ti.f64) -> ti.f64:
        return fget_ideal_activity_molar(mass_conc, self.molar_mass[None])

    # vectorised
    @ti.kernel
    def _activity_kernel(
        self,
        mass_concentration: ti.types.ndarray(dtype=ti.f64, ndim=1),
        result:             ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        mol = self.molar_mass[None]
        kget_ideal_activity_molar(mass_concentration, ti.static(mol), result)

    # public
    def activity(self, mass_concentration):
        if np.ndim(mass_concentration) == 0:
            return self._activity_func(float(mass_concentration))

        result = np.empty_like(mass_concentration, dtype=np.float64)
        self._activity_kernel(mass_concentration, result)
        return result


@ti.data_oriented
class ActivityIdealMass(_ActivityMixin):
    """Taichi drop-in for ActivityIdealMass (parameter-free)."""

    @ti.func
    def _activity_func(self, mass_conc: ti.f64) -> ti.f64:
        return fget_ideal_activity_mass(mass_conc)

    @ti.kernel
    def _activity_kernel(
        self,
        mass_concentration: ti.types.ndarray(dtype=ti.f64, ndim=1),
        result:             ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        kget_ideal_activity_mass(mass_concentration, result)

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

    @ti.func
    def _activity_func(self, mass_conc: ti.f64) -> ti.f64:
        return fget_ideal_activity_volume(mass_conc, self.density[None])

    @ti.kernel
    def _activity_kernel(
        self,
        mass_concentration: ti.types.ndarray(dtype=ti.f64, ndim=1),
        result:             ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        rho = self.density[None]
        kget_ideal_activity_volume(mass_concentration, ti.static(rho), result)

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

    # element-wise (rarely used – kept for completeness)
    @ti.func
    def _activity_func(self, mass_conc: ti.f64) -> ti.f64:      # scalar fall-back
        return fget_kappa_activity(
            mass_conc,
            self.kappa, self.density, self.molar_mass,
            self.water_index[None],
        )

    # vectorised
    @ti.kernel
    def _activity_kernel(
        self,
        mass_concentration: ti.types.ndarray(dtype=ti.f64, ndim=1),
        result:             ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        kget_kappa_activity(
            mass_concentration,
            self.kappa, self.density, self.molar_mass,
            self.water_index[None],
            result,
        )

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
def _make_vol(*a, **k):        return ActivityIdealVolume(*a, **k)

@register("ActivityKappaParameter", backend="taichi")
def _make_kappa(*a, **k):      return ActivityKappaParameter(*a, **k)
