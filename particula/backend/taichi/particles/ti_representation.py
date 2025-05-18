"""Taichi replacement for particula.particles.representation.ParticleRepresentation."""
import taichi as ti
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from particula.backend.dispatch_register import register

@ti.data_oriented
class _FieldIO:
    @ti.kernel
    def _assign_1d(self, field: ti.template(),
                   array: ti.types.ndarray(dtype=ti.f64, ndim=1)):
        for i in field:
            field[i] = array[i]

    @ti.kernel
    def _assign_2d(self, field: ti.template(),
                   array: ti.types.ndarray(dtype=ti.f64, ndim=2)):
        for i, j in field:
            field[i, j] = array[i, j]

    def from_numpy(self, field, array):
        if array.ndim == 1:
            self._assign_1d(field, array)
        elif array.ndim == 2:
            self._assign_2d(field, array)
        else:
            raise ValueError("Only 1-D/2-D supported")

_field_io = _FieldIO()

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
        _field_io.from_numpy(self.distribution,  distribution)
        _field_io.from_numpy(self.density,       density)
        _field_io.from_numpy(self.concentration, concentration)
        _field_io.from_numpy(self.charge,        charge)
        self.volume[None] = float(volume)

    # ───── getters identical to NumPy class (return NumPy views/copies) ─────
    def get_strategy(self, clone: bool = False):  return self.strategy if not clone else self.strategy.__copy__()
    def get_strategy_name(self) -> str:           return self.strategy.get_name()
    def get_activity(self, clone=False):          return self.activity if not clone else self.activity.__copy__()
    def get_activity_name(self) -> str:           return self.activity.get_name()
    def get_surface(self, clone=False):           return self.surface if not clone else self.surface.__copy__()
    def get_surface_name(self) -> str:            return self.surface.get_name()
    def get_distribution(self, clone=False):      # field → numpy
        distribution_array = self.distribution.to_numpy()
        return distribution_array.copy() if clone else distribution_array
    def get_density(self, clone=False):
        density_array = self.density.to_numpy()
        return density_array.copy() if clone else density_array

    # ───── density utilities ────────────────────────────────────────────
    def get_effective_density(self, clone: bool = False):
        """
        Return particle-wise effective density (mass-weighted species density).
        Mirrors particula.particles.representation.ParticleRepresentation.
        """
        densities = self.get_density()
        # single–species shortcut
        if isinstance(densities, float) or np.size(densities) == 1:
            eff = np.ones_like(self.get_species_mass()) * densities
        else:
            mass_total = self.get_mass()
            weighted_mass = np.sum(self.get_species_mass() * densities, axis=1)
            eff = np.divide(
                weighted_mass,
                mass_total,
                where=mass_total != 0,
                out=np.zeros_like(weighted_mass),
            )
        return eff.copy() if clone else eff

    def get_mean_effective_density(self) -> float:
        """
        Mean of the non-zero effective densities (0.0 if all zero).
        """
        effective = self.get_effective_density()
        effective = effective[effective != 0]
        if effective.size == 0:
            return 0.0
        return float(np.mean(effective))

    def __str__(self) -> str:
        return (
            f"Particle Representation:\n"
            f"\tStrategy: {self.get_strategy_name()}\n"
            f"\tActivity: {self.get_activity_name()}\n"
            f"\tSurface: {self.get_surface_name()}\n"
            f"\tMass Concentration: "
            f"{self.get_mass_concentration():.3e} [kg/m^3]\n"
            f"\tNumber Concentration: "
            f"{self.get_total_concentration():.3e} [#/m^3]"
        )

    def get_concentration(self, clone=False):
        system_volume = self.volume[None]
        concentration_array = self.concentration.to_numpy() / system_volume
        return concentration_array.copy() if clone else concentration_array
    def get_total_concentration(self, clone=False):
        return np.sum(self.get_concentration(clone=clone))
    def get_charge(self, clone=False):
        charge_array = self.charge.to_numpy()
        return charge_array.copy() if clone else charge_array
    def get_volume(self, clone=False):       return float(self.volume[None])

    # ───── derived quantities (delegate to strategy) ─────
    def get_species_mass(self, clone=False):
        species_mass = self.strategy.get_species_mass(
            self.get_distribution(), self.get_density()
        )
        return species_mass.copy() if clone else species_mass
    def get_mass(self, clone=False):
        particle_mass = self.strategy.get_mass(
            self.get_distribution(), self.get_density()
        )
        return particle_mass.copy() if clone else particle_mass
    def get_radius(self, clone=False):
        particle_radius = self.strategy.get_radius(
            self.get_distribution(), self.get_density()
        )
        return particle_radius.copy() if clone else particle_radius
    def get_mass_concentration(self, clone: bool = False):
        mass_concentration = self.strategy.get_total_mass(
            self.get_distribution(),
            self.get_concentration(),
            self.get_density(),
        )
        return float(mass_concentration)  # scalar, copy not needed but flag accepted

    # ───── mutators (update Ti fields in-place) ─────
    def add_mass(self, added_mass: NDArray[np.float64]) -> None:
        distribution_array, _ = self.strategy.add_mass(
            self.get_distribution(),
            self.get_concentration(),
            self.get_density(),
            added_mass
        )
        _field_io.from_numpy(self.distribution, distribution_array)

    def add_concentration(
        self,
        added_concentration: NDArray[np.float64],
        added_distribution: Optional[NDArray[np.float64]] = None
    ) -> None:
        if added_distribution is None:
            added_distribution = self.get_distribution()
        distribution_array, concentration_array = self.strategy.add_concentration(
            self.get_distribution(),
            self.get_concentration(),
            added_distribution=added_distribution,
            added_concentration=added_concentration
        )
        _field_io.from_numpy(self.distribution, distribution_array)
        _field_io.from_numpy(self.concentration, concentration_array * self.volume[None])

    def collide_pairs(self, indices: NDArray[np.int64]) -> None:
        distribution_array, concentration_array = self.strategy.collide_pairs(
            self.get_distribution(),
            self.get_concentration(),      # per m³ passed to strategy
            self.get_density(),
            indices,
        )
        _field_io.from_numpy(self.distribution, distribution_array)
        _field_io.from_numpy(self.concentration, concentration_array * self.volume[None])
