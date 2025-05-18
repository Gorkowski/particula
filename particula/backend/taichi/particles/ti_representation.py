"""Taichi replacement for particula.particles.representation.ParticleRepresentation."""
import taichi as ti
import numpy as np
from numpy.typing import NDArray
from typing import Optional

@ti.data_oriented
class _FieldIO:
    """
    Helper for copying NumPy arrays into Taichi fields (1D/2D only).

    Supports assignment of 1D and 2D float64 arrays to Taichi fields.
    Used for efficient data transfer between NumPy and Taichi.

    Examples:
        ```py
        field_io = _FieldIO()
        field_io.from_numpy(ti_field, np_array)
        ```
    """

    @ti.kernel
    def _assign_1d(self, field: ti.template(),
                   array: ti.types.ndarray(dtype=ti.f64, ndim=1)):
        """
        Assign a 1D NumPy array to a Taichi field.

        Args:
            - field : Taichi field to assign to.
            - array : 1D NumPy array of float64.

        Returns:
            - None
        """
        for i in field:
            field[i] = array[i]

    @ti.kernel
    def _assign_2d(self, field: ti.template(),
                   array: ti.types.ndarray(dtype=ti.f64, ndim=2)):
        """
        Assign a 2D NumPy array to a Taichi field.

        Args:
            - field : Taichi field to assign to.
            - array : 2D NumPy array of float64.

        Returns:
            - None
        """
        for i, j in field:
            field[i, j] = array[i, j]

    def from_numpy(self, field, array):
        """
        Copy a NumPy array (1D or 2D) into a Taichi field.

        Args:
            - field : Taichi field to assign to.
            - array : NumPy array (1D or 2D) of float64.

        Returns:
            - None

        Raises:
            - ValueError : If array is not 1D or 2D.
        """
        if array.ndim == 1:
            self._assign_1d(field, array)
        elif array.ndim == 2:
            self._assign_2d(field, array)
        else:
            raise ValueError("Only 1-D/2-D supported")

_field_io = _FieldIO()

@ti.data_oriented
class TiParticleRepresentation:
    """
    Taichi-based particle representation for multiphase systems.

    This class manages particle properties and their evolution using Taichi
    fields for high-performance simulation. It mirrors the interface of
    particula.particles.representation.ParticleRepresentation, but with
    Taichi-backed storage and operations.

    Attributes:
        - strategy : Particle strategy object (must support Taichi ops).
        - activity : Particle activity object.
        - surface : Particle surface object.
        - distribution : Taichi field, particle species distribution.
        - density : Taichi field, species densities.
        - concentration : Taichi field, particle concentrations.
        - charge : Taichi field, particle charges.
        - volume : Taichi field, system volume (scalar).

    Methods:
        - get_strategy, get_strategy_name
        - get_activity, get_activity_name
        - get_surface, get_surface_name
        - get_distribution, get_density, get_concentration, get_total_concentration
        - get_charge, get_volume
        - get_species_mass, get_mass, get_radius, get_mass_concentration
        - get_effective_density, get_mean_effective_density
        - add_mass, add_concentration, collide_pairs

    Examples:
        ```py
        rep = TiParticleRepresentation(
            strategy, activity, surface,
            distribution, density, concentration, charge, volume=1.0
        )
        mass = rep.get_mass()
        rep.add_mass(np.array([1.0, 2.0, 3.0]))
        ```

    Unicode equation for effective density:
        ρₑ = Σ mᵢ ρᵢ ⁄ mₜ
        where mᵢ is species mass, ρᵢ is species density, mₜ is total mass.

    """

    def __init__(self, strategy, activity, surface,
                 distribution: NDArray[np.float64],
                 density: NDArray[np.float64],
                 concentration: NDArray[np.float64],
                 charge: NDArray[np.float64], volume: float = 1.0):
        self.strategy, self.activity, self.surface = strategy, activity, surface
        self.distribution  = ti.field(ti.f64, shape=distribution.shape)
        self.density       = ti.field(ti.f64, shape=density.shape)
        self.concentration = ti.field(ti.f64, shape=concentration.shape)
        self.charge        = ti.field(ti.f64, shape=charge.shape)
        self.volume        = ti.field(ti.f64, shape=())
        _field_io.from_numpy(self.distribution,  distribution)
        _field_io.from_numpy(self.density,       density)
        _field_io.from_numpy(self.concentration, concentration)
        _field_io.from_numpy(self.charge,        charge)
        self.volume[None] = float(volume)

    # ───── getters identical to NumPy class (return NumPy views/copies) ─────
    def get_strategy(self, clone: bool = False):
        """
        Return the strategy instance (optionally shallow-copied).

        Args:
            - clone : If True, return a copy.

        Returns:
            - strategy : Strategy object.
        """
        return (
            self.strategy
            if not clone
            else self.strategy.__copy__()  # type: ignore[attr-defined]
        )

    def get_strategy_name(self) -> str:
        """
        Return the name of the strategy.

        Returns:
            - str : Name of the strategy.
        """
        return (
            self.strategy.get_name()
        )

    def get_activity(self, clone: bool = False):
        """
        Return the activity instance (optionally shallow-copied).

        Args:
            - clone : If True, return a copy.

        Returns:
            - activity : Activity object.
        """
        return (
            self.activity
            if not clone
            else self.activity.__copy__()  # type: ignore[attr-defined]
        )

    def get_activity_name(self) -> str:
        """
        Return the name of the activity.

        Returns:
            - str : Name of the activity.
        """
        return (
            self.activity.get_name()
        )

    def get_surface(self, clone: bool = False):
        """
        Return the surface instance (optionally shallow-copied).

        Args:
            - clone : If True, return a copy.

        Returns:
            - surface : Surface object.
        """
        return (
            self.surface
            if not clone
            else self.surface.__copy__()  # type: ignore[attr-defined]
        )

    def get_surface_name(self) -> str:
        """
        Return the name of the surface.

        Returns:
            - str : Name of the surface.
        """
        return (
            self.surface.get_name()
        )

    def get_distribution(self, clone: bool = False):
        """
        Return the particle species distribution as a NumPy array.

        Args:
            - clone : If True, return a copy.

        Returns:
            - np.ndarray : Distribution array.
        """
        distribution_array = self.distribution.to_numpy()
        return distribution_array.copy() if clone else distribution_array

    def get_density(self, clone: bool = False):
        """
        Return the species densities as a NumPy array.

        Args:
            - clone : If True, return a copy.

        Returns:
            - np.ndarray : Density array.
        """
        density_array = self.density.to_numpy()
        return density_array.copy() if clone else density_array

    # ───── density utilities ────────────────────────────────────────────
    def get_effective_density(self, clone: bool = False):
        """
        Return particle-wise effective density (mass-weighted species density).

        Mirrors particula.particles.representation.ParticleRepresentation.

        Returns:
            - np.ndarray : Effective density array.
        """
        densities = self.get_density()
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

        Returns:
            - float : Mean effective density.
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

    def get_concentration(self, clone: bool = False):
        """
        Return the particle concentration as a NumPy array.

        Args:
            - clone : If True, return a copy.

        Returns:
            - np.ndarray : Concentration array.
        """
        system_volume = self.volume[None]
        concentration_array = self.concentration.to_numpy() / system_volume
        return concentration_array.copy() if clone else concentration_array

    def get_total_concentration(self, clone: bool = False):
        """
        Return the total particle concentration (sum over all particles).

        Args:
            - clone : If True, return a copy.

        Returns:
            - float : Total concentration.
        """
        return np.sum(self.get_concentration(clone=clone))

    def get_charge(self, clone: bool = False):
        """
        Return the particle charge as a NumPy array.

        Args:
            - clone : If True, return a copy.

        Returns:
            - np.ndarray : Charge array.
        """
        charge_array = self.charge.to_numpy()
        return charge_array.copy() if clone else charge_array

    def get_volume(self, clone: bool = False):
        """
        Return the system volume.

        Args:
            - clone : Ignored (for API compatibility).

        Returns:
            - float : System volume.
        """
        return float(self.volume[None])

    # ───── derived quantities (delegate to strategy) ─────
    def get_species_mass(self, clone: bool = False):
        """
        Return the mass of each species for each particle.

        Args:
            - clone : If True, return a copy.

        Returns:
            - np.ndarray : Species mass array.
        """
        species_mass = self.strategy.get_species_mass(
            self.get_distribution(), self.get_density()
        )
        return species_mass.copy() if clone else species_mass

    def get_mass(self, clone: bool = False):
        """
        Return the total mass for each particle.

        Args:
            - clone : If True, return a copy.

        Returns:
            - np.ndarray : Particle mass array.
        """
        particle_mass = self.strategy.get_mass(
            self.get_distribution(), self.get_density()
        )
        return particle_mass.copy() if clone else particle_mass

    def get_radius(self, clone: bool = False):
        """
        Return the radius for each particle.

        Args:
            - clone : If True, return a copy.

        Returns:
            - np.ndarray : Particle radius array.
        """
        particle_radius = self.strategy.get_radius(
            self.get_distribution(), self.get_density()
        )
        return particle_radius.copy() if clone else particle_radius

    def get_mass_concentration(self, clone: bool = False):
        """
        Return the total mass concentration (scalar).

        Args:
            - clone : Ignored (for API compatibility).

        Returns:
            - float : Mass concentration.
        """
        mass_concentration = self.strategy.get_total_mass(
            self.get_distribution(),
            self.get_concentration(),
            self.get_density(),
        )
        return float(mass_concentration)

    # ───── mutators (update Ti fields in-place) ─────
    def add_mass(self, added_mass: NDArray[np.float64]) -> None:
        """
        Add mass to the particle distribution in-place.

        Args:
            - added_mass : Array of mass to add per particle.

        Returns:
            - None
        """
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
        """
        Add concentration (and optionally distribution) to the system in-place.

        Args:
            - added_concentration : Array of concentration to add.
            - added_distribution : Optional distribution array.

        Returns:
            - None
        """
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
        """
        Collide pairs of particles, updating distribution and concentration.

        Args:
            - indices : Array of particle index pairs to collide.

        Returns:
            - None
        """
        distribution_array, concentration_array = self.strategy.collide_pairs(
            self.get_distribution(),
            self.get_concentration(),
            self.get_density(),
            indices,
        )
        _field_io.from_numpy(self.distribution, distribution_array)
        _field_io.from_numpy(self.concentration, concentration_array * self.volume[None])
