"""Taichi replacement for particula.particles.representation.ParticleRepresentation."""
import taichi as ti
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from particula.backend.dispatch_register import register

@ti.data_oriented
class _FieldIO:                                            # ← tiny helper‐mixin
    @staticmethod @ti.kernel
    def _assign_1d(f: ti.template(), a: ti.types.ndarray(dtype=ti.f64, ndim=1)):
        for i in f: f[i] = a[i]
    @staticmethod @ti.kernel
    def _assign_2d(f: ti.template(), a: ti.types.ndarray(dtype=ti.f64, ndim=2)):
        for i, j in f: f[i, j] = a[i, j]
    @classmethod
    def from_numpy(cls, fld, arr):
        if arr.ndim == 1: cls._assign_1d(fld, arr)
        elif arr.ndim == 2: cls._assign_2d(fld, arr)
        else: raise ValueError("Only 1-D/2-D supported")

@register("ParticleRepresentation", backend="taichi")      # backend factory-hook
@ti.data_oriented
class TiParticleRepresentation:                            # public Taichi class
    def __init__(self, strategy, activity, surface,
                 distribution: NDArray[np.float64],
                 density: NDArray[np.float64],
                 concentration: NDArray[np.float64],
                 charge: NDArray[np.float64], volume: float = 1.0):
        # collaborators are already Taichi versions
        self.strategy, self.activity, self.surface = strategy, activity, surface
        # allocate persistent fields
        self.distribution  = ti.field(ti.f64, shape=distribution.shape)
        self.density       = ti.field(ti.f64, shape=density.shape)
        self.concentration = ti.field(ti.f64, shape=concentration.shape)
        self.charge        = ti.field(ti.f64, shape=charge.shape)
        self.volume        = ti.field(ti.f64, shape=())
        # copy initial values
        _FieldIO.from_numpy(self.distribution,  distribution)
        _FieldIO.from_numpy(self.density,       density)
        _FieldIO.from_numpy(self.concentration, concentration)
        _FieldIO.from_numpy(self.charge,        charge)
        self.volume[None] = float(volume)

    # ───── getters identical to NumPy class (return NumPy views/copies) ─────
    def get_strategy(self, clone: bool = False):  return self.strategy if not clone else self.strategy.__copy__()
    def get_strategy_name(self) -> str:           return self.strategy.get_name()
    def get_activity(self, clone=False):          return self.activity if not clone else self.activity.__copy__()
    def get_activity_name(self) -> str:           return self.activity.get_name()
    def get_surface(self, clone=False):           return self.surface if not clone else self.surface.__copy__()
    def get_surface_name(self) -> str:            return self.surface.get_name()
    def get_distribution(self, clone=False):      # field → numpy
        arr = self.distribution.to_numpy();  return arr.copy() if clone else arr
    def get_density(self, clone=False):
        arr = self.density.to_numpy();       return arr.copy() if clone else arr
    def get_concentration(self, clone=False):
        vol = self.volume[None]
        arr = self.concentration.to_numpy() / vol
        return arr.copy() if clone else arr
    def get_total_concentration(self, clone=False):
        return np.sum(self.get_concentration(clone=clone))
    def get_charge(self, clone=False):
        arr = self.charge.to_numpy();        return arr.copy() if clone else arr
    def get_volume(self, clone=False):       return float(self.volume[None])

    # ───── derived quantities (delegate to strategy) ─────
    def get_species_mass(self, clone=False):
        res = self.strategy.get_species_mass(self.get_distribution(), self.get_density())
        return res.copy() if clone else res
    def get_mass(self, clone=False):
        res = self.strategy.get_mass(self.get_distribution(), self.get_density())
        return res.copy() if clone else res
    def get_radius(self, clone=False):
        res = self.strategy.get_radius(self.get_distribution(), self.get_density())
        return res.copy() if clone else res
    def get_mass_concentration(self, clone=False):
        m_c = self.strategy.get_total_mass(self.get_distribution(),
                                           self.get_concentration(),
                                           self.get_density())
        return float(m_c)

    # ───── mutators (update Ti fields in-place) ─────
    def add_mass(self, added_mass: NDArray[np.float64]) -> None:
        dist, _ = self.strategy.add_mass(self.get_distribution(),
                                         self.get_concentration(),
                                         self.get_density(),
                                         added_mass)
        _FieldIO.from_numpy(self.distribution, dist)

    def add_concentration(self, added_conc: NDArray[np.float64],
                          added_dist: Optional[NDArray[np.float64]] = None) -> None:
        if added_dist is None: added_dist = self.get_distribution()
        dist, conc = self.strategy.add_concentration(self.get_distribution(),
                                                     self.get_concentration(),
                                                     added_distribution=added_dist,
                                                     added_concentration=added_conc)
        _FieldIO.from_numpy(self.distribution,  dist)
        _FieldIO.from_numpy(self.concentration, conc * self.volume[None])

    def collide_pairs(self, indices: NDArray[np.int64]) -> None:
        dist, conc = self.strategy.collide_pairs(self.get_distribution(),
                                                 self.get_concentration(),
                                                 self.get_density(), indices)
        _FieldIO.from_numpy(self.distribution,  dist)
        _FieldIO.from_numpy(self.concentration, conc)
