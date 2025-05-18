"""
Taichi-based implementations of particula.particles.distribution_strategies.*

This module provides Taichi-accelerated drop-in replacements for the
distribution strategy classes in the particula.particles.distribution_strategies
Python module. These classes are designed for high-performance particle
distribution calculations, mirroring the public API of their Python
counterparts. All classes inherit from a shared mix-in for volume/radius
conversion helpers.

Classes:
    - _DistributionMixin: Shared Taichi helpers for volume/radius conversion.
    - MassBasedMovingBin: Mass-based moving bin strategy (Taichi).
    - RadiiBasedMovingBin: Radii-based moving bin strategy (Taichi).
    - SpeciatedMassMovingBin: Speciated mass moving bin strategy (Taichi).
    - ParticleResolvedSpeciatedMass: Particle-resolved speciated mass strategy.

Examples:
    ```py
    from particula.backend.taichi.particles.ti_distribution_strategies import (
        MassBasedMovingBin
    )
    import numpy as np

    distribution = np.array([1.0, 2.0, 3.0])
    density = np.array([1000.0, 1000.0, 1000.0])
    strategy = MassBasedMovingBin()
    radii = strategy.get_radius(distribution, density)
    print(radii)
    ```

References:
    - particula.particles.distribution_strategies (Python backend)
"""

import taichi as ti
import numpy as np
from numpy.typing import NDArray
from particula.backend.dispatch_register import register

ti.init(arch=ti.cpu, default_fp=ti.f64)     # guard against forgotten init

# ─── mix-in holding helpers used in several subclasses ─────────────────
@ti.data_oriented
class _DistributionMixin:
    """
    Mixin providing Taichi helpers for volume/radius conversions.

    This class supplies static Taichi functions for converting between
    particle volume and radius. It is intended to be inherited by
    distribution strategy classes.

    Methods:
        - _volume_to_radius: Convert volume [m³] to radius [m].
        - _radius_to_volume: Convert radius [m] to volume [m³].
    """

    @staticmethod
    @ti.func
    def _volume_to_radius(volume: ti.f64) -> ti.f64:
        """Convert particle volume [m³] to radius [m]."""
        return ti.pow(3.0 * volume / (4.0 * ti.math.pi), 1.0 / 3.0)

    @staticmethod
    @ti.func
    def _radius_to_volume(radius: ti.f64) -> ti.f64:
        """Convert particle radius [m] to volume [m³]."""
        return 4.0 / 3.0 * ti.math.pi * radius ** 3

# ──────────────────────────────────────────────────────────────────────
@register("MassBasedMovingBin", backend="taichi")
@ti.data_oriented
class MassBasedMovingBin(_DistributionMixin):
    """
    Taichi implementation of the mass-based moving bin strategy.

    This class is a drop-in Taichi-accelerated replacement for
    particula.particles.distribution_strategies.MassBasedMovingBin.
    It mirrors the public API of the Python backend, providing
    high-performance methods for mass-based particle bin calculations.

    Attributes:
        - Inherits volume/radius helpers from _DistributionMixin.

    Methods:
        - get_name: Return the class name.
        - get_species_mass: Compute species-level mass.
        - get_mass: Compute total mass per bin/particle.
        - get_total_mass: Compute total system mass.
        - get_radius: Compute particle/bin radii.
        - add_mass: Add mass to the distribution.
        - add_concentration: Add concentration to the distribution.
        - collide_pairs: Not implemented for this strategy.

    Examples:
        ```py
        strategy = MassBasedMovingBin()
        radii = strategy.get_radius(distribution, density)
        ```

    References:
        - particula.particles.distribution_strategies.MassBasedMovingBin
    """
    def __init__(self):
        pass

    # ───────────────────── ti.func helper layer ──────────────────────
    # These tiny per-element helpers expose the class’ core maths so
    # that *other* kernels may `import` the class and re-use them via
    # dependency-injection without having to call the numpy wrappers.

    @ti.func
    def fget_species_mass(
        self, distribution_mass: ti.f64, density: ti.f64
    ) -> ti.f64:
        """Return species-level mass [kg] (identity for this strategy)."""
        return distribution_mass

    @ti.func
    def fget_particle_mass(self, species_masses: ti.template()) -> ti.f64:
        """Return sum of species masses for one particle/bin [kg]."""
        total = 0.0
        for k in ti.static(range(species_masses.n)):
            total += species_masses[k]
        return total

    @ti.func
    def fget_particle_radius(
        self, particle_mass: ti.f64, particle_density: ti.f64
    ) -> ti.f64:
        """Return radius [m] from total mass and density."""
        volume = particle_mass / particle_density
        return self._volume_to_radius(volume)

    @ti.func
    def fget_updated_mass(
        self,
        current_mass: ti.f64,
        concentration: ti.f64,
        added_mass: ti.f64,
    ) -> ti.f64:
        """Return updated mass [kg] after adding added_mass."""
        if concentration != 0.0:
            new_mass = (current_mass * concentration + added_mass) / concentration
            return ti.max(new_mass, 0.0)
        else:
            return 0.0

    @ti.func
    def fget_total_mass(
        self, particle_mass: ti.f64, concentration: ti.f64
    ) -> ti.f64:
        """Return total mass [kg] for a single particle/bin."""
        return particle_mass * concentration

    @ti.func
    def fget_merge_concentration(
        self,
        concentration_old: ti.f64,
        concentration_added: ti.f64,
    ) -> ti.f64:
        """Return merged concentration (sum of old and added)."""
        return concentration_old + concentration_added

    def get_name(self) -> str:
        """
        Return the class name.

        Returns:
            - str : Name of the class.
        """
        return self.__class__.__name__

    @ti.func
    def _fget_species_mass(self, distribution, density):
        """Return species-level mass [kg] (identity for this strategy)."""
        return distribution

    @ti.kernel
    def _kget_species_mass_1d(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64, ndim=1),
        density:       ti.types.ndarray(dtype=ti.f64, ndim=1),
        result:        ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        for i in range(distribution.shape[0]):
            result[i] = self._fget_species_mass(distribution[i], density[i])

    @ti.kernel
    def _kget_species_mass_2d(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64, ndim=2),
        density:       ti.types.ndarray(dtype=ti.f64, ndim=2),
        result:        ti.types.ndarray(dtype=ti.f64, ndim=2),
    ):
        for i, j in ti.ndrange(distribution.shape[0], distribution.shape[1]):
            result[i, j] = self._fget_species_mass(
                distribution[i, j], density[i, j]
            )

    def get_species_mass(
        self,
        distribution: NDArray[np.float64],
        density: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute the species-level mass for each bin/particle.

        Arguments:
            - distribution : NDArray[np.float64]
                Mass distribution array [kg].
            - density : NDArray[np.float64]
                Density array [kg/m³].

        Returns:
            - NDArray[np.float64] : Species mass array [kg].
        """
        result = np.empty_like(distribution)
        if distribution.ndim == 1:
            self._kget_species_mass_1d(distribution, density, result)
        elif distribution.ndim == 2:
            self._kget_species_mass_2d(distribution, density, result)
        else:
            raise ValueError("Only 1-D or 2-D distributions are supported")
        return result

    def get_mass(self, distribution, density):
        """
        Compute the total mass for each bin/particle.

        Arguments:
            - distribution : NDArray[np.float64]
                Mass distribution array [kg].
            - density : NDArray[np.float64]
                Density array [kg/m³].

        Returns:
            - NDArray[np.float64] or float : Total mass per bin/particle [kg].
        """
        species_mass = self.get_species_mass(distribution, density)
        if distribution.ndim == 1:
            return species_mass
        return np.sum(species_mass, axis=1)

    def get_total_mass(self, distribution, concentration, density):
        """
        Compute the total system mass.

        Arguments:
            - distribution : NDArray[np.float64]
                Mass distribution array [kg].
            - concentration : NDArray[np.float64]
                Concentration array [1/m³].
            - density : NDArray[np.float64]
                Density array [kg/m³].

        Returns:
            - float : Total mass [kg].
        """
        return np.sum(self.get_mass(distribution, density) * concentration)

    def get_radius(self, distribution, density):
        """
        Compute the radius for each bin/particle.

        Arguments:
            - distribution : NDArray[np.float64]
                Mass distribution array [kg].
            - density : NDArray[np.float64]
                Density array [kg/m³].

        Returns:
            - NDArray[np.float64] : Radii [m].

        Examples:
            ```py
            strategy = MassBasedMovingBin()
            radii = strategy.get_radius(distribution, density)
            ```
        """
        # Calculate the volume of each particle from its mass and density,
        # then calculate the radius.
        volumes = distribution / density
        return (3 * volumes / (4 * np.pi)) ** (1 / 3)

    def add_mass(self, distribution, concentration, density, added_mass):
        """
        Add mass to the distribution, moving the bins.

        Arguments:
            - distribution : NDArray[np.float64]
                Mass distribution array [kg].
            - concentration : NDArray[np.float64]
                Concentration array [1/m³].
            - density : NDArray[np.float64]
                Density array [kg/m³].
            - added_mass : NDArray[np.float64]
                Mass to add [kg].

        Returns:
            - Tuple[NDArray, NDArray] : (updated distribution, concentration)
        """
        return (distribution + added_mass, concentration)

    def add_concentration(self, distribution, concentration,
                          added_distribution, added_concentration):
        """
        Add concentration to the distribution.

        Arguments:
            - distribution : NDArray[np.float64]
                Mass distribution array [kg].
            - concentration : NDArray[np.float64]
                Concentration array [1/m³].
            - added_distribution : NDArray[np.float64]
                Added mass distribution [kg].
            - added_concentration : NDArray[np.float64]
                Added concentration [1/m³].

        Returns:
            - Tuple[NDArray, NDArray] : (distribution, updated concentration)
        """
        if (distribution.shape != added_distribution.shape) and (
            np.allclose(distribution, added_distribution, rtol=1e-6)
        ):
            message = (
                "When adding concentration to MassBasedMovingBin, "
                "the distribution and added distribution should have the "
                "same elements. The current distribution shape is "
                f"{distribution.shape} and the added distribution shape is "
                f"{added_distribution.shape}."
            )
            raise ValueError(message)
        if concentration.shape != added_concentration.shape:
            message = (
                "When adding concentration to MassBasedMovingBin, "
                "the concentration and added concentration should have the "
                "same shape. The current concentration shape is "
                f"{concentration.shape} and the added concentration shape is "
                f"{added_concentration.shape}."
            )
            raise ValueError(message)
        concentration += added_concentration
        return (distribution, concentration)

    def collide_pairs(self, distribution, concentration, density, indices):
        """
        Not implemented for this strategy.

        Raises:
            - NotImplementedError : Always raised for this method.
        """
        message = (
            "Colliding pairs in MassBasedMovingBin is not physically "
            "meaningful, change dyanmic or particle strategy."
        )
        raise NotImplementedError(message)

# ──────────────────────────────────────────────────────────────────────
@register("RadiiBasedMovingBin", backend="taichi")
@ti.data_oriented
class RadiiBasedMovingBin(_DistributionMixin):
    """
    Taichi implementation of the radii-based moving bin strategy.

    This class is a drop-in Taichi-accelerated replacement for
    particula.particles.distribution_strategies.RadiiBasedMovingBin.
    It mirrors the public API of the Python backend, providing
    high-performance methods for radii-based particle bin calculations.

    Attributes:
        - Inherits volume/radius helpers from _DistributionMixin.

    Methods:
        - get_name: Return the class name.
        - get_species_mass: Compute species-level mass.
        - get_mass: Compute total mass per bin/particle.
        - get_total_mass: Compute total system mass.
        - get_radius: Return the radii (identity).
        - add_mass: Add mass to the distribution.
        - add_concentration: Add concentration to the distribution.
        - collide_pairs: Not implemented for this strategy.

    Examples:
        ```py
        strategy = RadiiBasedMovingBin()
        radii = strategy.get_radius(distribution, density)
        ```

    References:
        - particula.particles.distribution_strategies.RadiiBasedMovingBin
    """
    def __init__(self):
        pass

    def get_name(self) -> str:      # parity with python back-end
        return self.__class__.__name__

    @ti.func
    def _fget_species_mass(self, distribution, density):
        """Return species-level mass [kg] for a given radius and density."""
        return 4.0 / 3.0 * ti.math.pi * distribution ** 3 * density

    @ti.kernel
    def _kget_species_mass_1d(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64, ndim=1),
        density:       ti.types.ndarray(dtype=ti.f64, ndim=1),
        result:        ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        for i in range(distribution.shape[0]):
            result[i] = self._fget_species_mass(distribution[i], density[i])

    @ti.kernel
    def _kget_species_mass_2d(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64, ndim=2),
        density:       ti.types.ndarray(dtype=ti.f64, ndim=2),
        result:        ti.types.ndarray(dtype=ti.f64, ndim=2),
    ):
        for i, j in ti.ndrange(distribution.shape[0], distribution.shape[1]):
            result[i, j] = self._fget_species_mass(
                distribution[i, j], density[i, j]
            )

    def get_species_mass(
        self,
        distribution: NDArray[np.float64],
        density: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute the species-level mass for each bin/particle.

        Arguments:
            - distribution : NDArray[np.float64]
                Radii distribution array [m].
            - density : NDArray[np.float64]
                Density array [kg/m³].

        Returns:
            - NDArray[np.float64] : Species mass array [kg].
        """
        result = np.empty_like(distribution)
        if distribution.ndim == 1:
            self._kget_species_mass_1d(distribution, density, result)
        elif distribution.ndim == 2:
            self._kget_species_mass_2d(distribution, density, result)
        else:
            raise ValueError("Only 1-D or 2-D distributions are supported")
        return result

    def get_mass(self, distribution, density):
        """
        Compute the total mass for each bin/particle.

        Arguments:
            - distribution : NDArray[np.float64]
                Radii distribution array [m].
            - density : NDArray[np.float64]
                Density array [kg/m³].

        Returns:
            - NDArray[np.float64] or float : Total mass per bin/particle [kg].
        """
        species_mass = self.get_species_mass(distribution, density)
        if distribution.ndim == 1:
            return species_mass
        return np.sum(species_mass, axis=1)

    def get_total_mass(self, distribution, concentration, density):
        """
        Compute the total system mass.

        Arguments:
            - distribution : NDArray[np.float64]
                Radii distribution array [m].
            - concentration : NDArray[np.float64]
                Concentration array [1/m³].
            - density : NDArray[np.float64]
                Density array [kg/m³].

        Returns:
            - float : Total mass [kg].
        """
        return np.sum(self.get_mass(distribution, density) * concentration)

    def get_radius(self, distribution, density):
        """
        Return the radii (identity).

        Arguments:
            - distribution : NDArray[np.float64]
                Radii distribution array [m].
            - density : NDArray[np.float64]
                Density array [kg/m³] (unused).

        Returns:
            - NDArray[np.float64] : Radii [m].
        """
        return distribution

    def add_mass(self, distribution, concentration, density, added_mass):
        """
        Add mass to the distribution, updating the radii.

        Arguments:
            - distribution : NDArray[np.float64]
                Radii distribution array [m].
            - concentration : NDArray[np.float64]
                Concentration array [1/m³].
            - density : NDArray[np.float64]
                Density array [kg/m³].
            - added_mass : NDArray[np.float64]
                Mass to add [kg].

        Returns:
            - Tuple[NDArray, NDArray] : (updated radii, concentration)
        """
        mass_per_particle = np.where(
            concentration > 0, added_mass / concentration, 0
        )
        initial_volumes = (4 / 3) * np.pi * np.power(distribution, 3)
        new_volumes = initial_volumes + mass_per_particle / density
        new_radii = np.power(3 * new_volumes / (4 * np.pi), 1 / 3)
        return (new_radii, concentration)

    def add_concentration(self, distribution, concentration,
                          added_distribution, added_concentration):
        """
        Add concentration to the distribution.

        Arguments:
            - distribution : NDArray[np.float64]
                Radii distribution array [m].
            - concentration : NDArray[np.float64]
                Concentration array [1/m³].
            - added_distribution : NDArray[np.float64]
                Added radii distribution [m].
            - added_concentration : NDArray[np.float64]
                Added concentration [1/m³].

        Returns:
            - Tuple[NDArray, NDArray] : (distribution, updated concentration)
        """
        if (distribution.shape != added_distribution.shape) and (
            np.allclose(distribution, added_distribution, rtol=1e-6)
        ):
            message = (
                "When adding concentration to RadiiBasedMovingBin, "
                "the distribution and added distribution should have the "
                "same elements. The current distribution shape is "
                f"{distribution.shape} and the added distribution shape is "
                f"{added_distribution.shape}."
            )
            raise ValueError(message)
        if concentration.shape != added_concentration.shape:
            message = (
                "When adding concentration to RadiiBasedMovingBin, "
                "the concentration and added concentration should have the "
                "same shape. The current concentration shape is "
                f"{concentration.shape} and the added concentration shape is "
                f"{added_concentration.shape}."
            )
            raise ValueError(message)
        concentration += added_concentration
        return (distribution, concentration)

    def collide_pairs(self, distribution, concentration, density, indices):
        """
        Not implemented for this strategy.

        Raises:
            - NotImplementedError : Always raised for this method.
        """
        message = (
            "Colliding pairs in RadiiBasedMovingBin is not physically "
            "meaningful, change dyanmic or particle strategy."
        )
        raise NotImplementedError(message)

# ──────────────────────────────────────────────────────────────────────
@register("SpeciatedMassMovingBin", backend="taichi")
@ti.data_oriented
class SpeciatedMassMovingBin(_DistributionMixin):
    """
    Taichi implementation of the speciated mass moving bin strategy.

    This class is a drop-in Taichi-accelerated replacement for
    particula.particles.distribution_strategies.SpeciatedMassMovingBin.
    It mirrors the public API of the Python backend, providing
    high-performance methods for speciated mass bin calculations.

    Attributes:
        - Inherits volume/radius helpers from _DistributionMixin.

    Methods:
        - get_name: Return the class name.
        - get_species_mass: Compute species-level mass.
        - get_mass: Compute total mass per bin/particle.
        - get_total_mass: Compute total system mass.
        - get_radius: Compute particle/bin radii.
        - add_mass: Add mass to the distribution.
        - add_concentration: Add concentration to the distribution.
        - collide_pairs: Not implemented for this strategy.

    Examples:
        ```py
        strategy = SpeciatedMassMovingBin()
        radii = strategy.get_radius(distribution, density)
        ```

    References:
        - particula.particles.distribution_strategies.SpeciatedMassMovingBin
    """
    def __init__(self):
        pass

    def get_name(self) -> str:      # parity with python back-end
        return self.__class__.__name__

    @ti.func
    def _fget_species_mass(self, distribution, density):
        """Return species-level mass [kg] (identity for this strategy)."""
        return distribution

    @ti.kernel
    def _kget_species_mass_1d(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64, ndim=1),
        density:       ti.types.ndarray(dtype=ti.f64, ndim=1),
        result:        ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        for i in range(distribution.shape[0]):
            result[i] = self._fget_species_mass(distribution[i], density[i])

    @ti.kernel
    def _kget_species_mass_2d(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64, ndim=2),
        density:       ti.types.ndarray(dtype=ti.f64, ndim=2),
        result:        ti.types.ndarray(dtype=ti.f64, ndim=2),
    ):
        for i, j in ti.ndrange(distribution.shape[0], distribution.shape[1]):
            result[i, j] = self._fget_species_mass(
                distribution[i, j], density[i, j]
            )

    def get_species_mass(
        self,
        distribution: NDArray[np.float64],
        density: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute the species-level mass for each bin/particle.

        Arguments:
            - distribution : NDArray[np.float64]
                Mass distribution array [kg].
            - density : NDArray[np.float64]
                Density array [kg/m³].

        Returns:
            - NDArray[np.float64] : Species mass array [kg].
        """
        result = np.empty_like(distribution)
        if distribution.ndim == 1:
            self._kget_species_mass_1d(distribution, density, result)
        elif distribution.ndim == 2:
            self._kget_species_mass_2d(distribution, density, result)
        else:
            raise ValueError("Only 1-D or 2-D distributions are supported")
        return result

    def get_mass(self, distribution, density):
        """
        Compute the total mass for each bin/particle.

        Arguments:
            - distribution : NDArray[np.float64]
                Mass distribution array [kg].
            - density : NDArray[np.float64]
                Density array [kg/m³].

        Returns:
            - NDArray[np.float64] or float : Total mass per bin/particle [kg].
        """
        species_mass = self.get_species_mass(distribution, density)
        if distribution.ndim == 1:
            return species_mass
        return np.sum(species_mass, axis=1)

    def get_total_mass(self, distribution, concentration, density):
        """
        Compute the total system mass.

        Arguments:
            - distribution : NDArray[np.float64]
                Mass distribution array [kg].
            - concentration : NDArray[np.float64]
                Concentration array [1/m³].
            - density : NDArray[np.float64]
                Density array [kg/m³].

        Returns:
            - float : Total mass [kg].
        """
        return np.sum(self.get_mass(distribution, density) * concentration)

    def get_radius(self, distribution, density):
        """
        Compute the radius for each bin/particle.

        Arguments:
            - distribution : NDArray[np.float64]
                Mass distribution array [kg].
            - density : NDArray[np.float64]
                Density array [kg/m³].

        Returns:
            - NDArray[np.float64] : Radii [m].
        """
        volumes = np.sum(distribution / density, axis=1)
        return (3 * volumes / (4 * np.pi)) ** (1 / 3)

    def add_mass(self, distribution, concentration, density, added_mass):
        """
        Add mass to the distribution.

        Arguments:
            - distribution : NDArray[np.float64]
                Mass distribution array [kg].
            - concentration : NDArray[np.float64]
                Concentration array [1/m³].
            - density : NDArray[np.float64]
                Density array [kg/m³].
            - added_mass : NDArray[np.float64]
                Mass to add [kg].

        Returns:
            - Tuple[NDArray, NDArray] : (updated distribution, concentration)
        """
        if distribution.ndim == 2:
            concentration_expand = concentration[:, np.newaxis]
        else:
            concentration_expand = concentration
        mass_per_particle = np.where(
            concentration_expand > 0, added_mass / concentration_expand, 0
        )
        new_distribution = np.maximum(distribution + mass_per_particle, 0)
        return new_distribution, concentration

    def add_concentration(self, distribution, concentration,
                          added_distribution, added_concentration):
        """
        Add concentration to the distribution.

        Arguments:
            - distribution : NDArray[np.float64]
                Mass distribution array [kg].
            - concentration : NDArray[np.float64]
                Concentration array [1/m³].
            - added_distribution : NDArray[np.float64]
                Added mass distribution [kg].
            - added_concentration : NDArray[np.float64]
                Added concentration [1/m³].

        Returns:
            - Tuple[NDArray, NDArray] : (distribution, updated concentration)
        """
        if (distribution.shape != added_distribution.shape) and (
            np.allclose(distribution, added_distribution, rtol=1e-6)
        ):
            message = (
                "When adding concentration to SpeciatedMassMovingBin, "
                "the distribution and added distribution should have the "
                "same elements. The current distribution shape is "
                f"{distribution.shape} and the added distribution shape is "
                f"{added_distribution.shape}."
            )
            raise ValueError(message)
        if concentration.shape != added_concentration.shape:
            message = (
                "When adding concentration to SpeciatedMassMovingBin, "
                "the concentration and added concentration should have the "
                "same shape. The current concentration shape is "
                f"{concentration.shape} and the added concentration shape is "
                f"{added_concentration.shape}."
            )
            raise ValueError(message)
        concentration += added_concentration
        return (distribution, concentration)

    def collide_pairs(self, distribution, concentration, density, indices):
        """
        Not implemented for this strategy.

        Raises:
            - NotImplementedError : Always raised for this method.
        """
        message = (
            "Colliding pairs in SpeciatedMassMovingBin is not physically "
            "meaningful, change dyanmic or particle strategy."
        )
        import logging
        logger = logging.getLogger("particula")
        logger.warning(message)
        raise NotImplementedError(message)

# ──────────────────────────────────────────────────────────────────────
@register("ParticleResolvedSpeciatedMass", backend="taichi")
@ti.data_oriented
class ParticleResolvedSpeciatedMass(_DistributionMixin):
    """
    Taichi implementation of the particle-resolved speciated mass strategy.

    This class is a drop-in Taichi-accelerated replacement for
    particula.particles.distribution_strategies.ParticleResolvedSpeciatedMass.
    It mirrors the public API of the Python backend, providing
    high-performance methods for particle-resolved speciated mass
    calculations.

    Attributes:
        - Inherits volume/radius helpers from _DistributionMixin.

    Methods:
        - get_name: Return the class name.
        - get_species_mass: Compute species-level mass.
        - get_mass: Compute total mass per bin/particle.
        - get_total_mass: Compute total system mass.
        - get_radius: Compute particle/bin radii.
        - add_mass: Add mass to the distribution.
        - add_concentration: Add concentration to the distribution.
        - collide_pairs: Merge pairs of particles/bins.

    Examples:
        ```py
        strategy = ParticleResolvedSpeciatedMass()
        radii = strategy.get_radius(distribution, density)
        ```

    References:
        - particula.particles.distribution_strategies.ParticleResolvedSpeciatedMass
    """
    def __init__(self):
        pass

    def get_name(self) -> str:      # parity with python back-end
        return self.__class__.__name__

    @ti.func
    def _fget_species_mass(self, distribution, density):
        """Return species-level mass [kg] (identity for this strategy)."""
        return distribution

    @ti.kernel
    def _kget_species_mass_1d(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64, ndim=1),
        density:       ti.types.ndarray(dtype=ti.f64, ndim=1),
        result:        ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        for i in range(distribution.shape[0]):
            result[i] = self._fget_species_mass(distribution[i], density[i])

    @ti.kernel
    def _kget_species_mass_2d(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64, ndim=2),
        density:       ti.types.ndarray(dtype=ti.f64, ndim=2),
        result:        ti.types.ndarray(dtype=ti.f64, ndim=2),
    ):
        for i, j in ti.ndrange(distribution.shape[0], distribution.shape[1]):
            result[i, j] = self._fget_species_mass(
                distribution[i, j], density[i, j]
            )

    def get_species_mass(
        self,
        distribution: NDArray[np.float64],
        density: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute the species-level mass for each bin/particle.

        Arguments:
            - distribution : NDArray[np.float64]
                Mass distribution array [kg].
            - density : NDArray[np.float64]
                Density array [kg/m³].

        Returns:
            - NDArray[np.float64] : Species mass array [kg].
        """
        result = np.empty_like(distribution)
        if distribution.ndim == 1:
            self._kget_species_mass_1d(distribution, density, result)
        elif distribution.ndim == 2:
            self._kget_species_mass_2d(distribution, density, result)
        else:
            raise ValueError("Only 1-D or 2-D distributions are supported")
        return result

    def get_mass(self, distribution, density):
        """
        Compute the total mass for each bin/particle.

        Arguments:
            - distribution : NDArray[np.float64]
                Mass distribution array [kg].
            - density : NDArray[np.float64]
                Density array [kg/m³].

        Returns:
            - NDArray[np.float64] or float : Total mass per bin/particle [kg].
        """
        species_mass = self.get_species_mass(distribution, density)
        if distribution.ndim == 1:
            return species_mass
        return np.sum(species_mass, axis=1)

    def get_total_mass(self, distribution, concentration, density):
        """
        Compute the total system mass.

        Arguments:
            - distribution : NDArray[np.float64]
                Mass distribution array [kg].
            - concentration : NDArray[np.float64]
                Concentration array [1/m³].
            - density : NDArray[np.float64]
                Density array [kg/m³].

        Returns:
            - float : Total mass [kg].
        """
        return np.sum(self.get_mass(distribution, density) * concentration)

    def get_radius(self, distribution, density):
        """
        Compute the radius for each bin/particle.

        Arguments:
            - distribution : NDArray[np.float64]
                Mass distribution array [kg].
            - density : NDArray[np.float64]
                Density array [kg/m³].

        Returns:
            - NDArray[np.float64] : Radii [m].
        """
        if distribution.ndim == 1:
            volumes = distribution / density
        else:
            volumes = np.sum(distribution / density, axis=1)
        return (3 * volumes / (4 * np.pi)) ** (1 / 3)

    def add_mass(self, distribution, concentration, density, added_mass):
        """
        Add mass to the distribution.

        Arguments:
            - distribution : NDArray[np.float64]
                Mass distribution array [kg].
            - concentration : NDArray[np.float64]
                Concentration array [1/m³].
            - density : NDArray[np.float64]
                Density array [kg/m³].
            - added_mass : NDArray[np.float64]
                Mass to add [kg].

        Returns:
            - Tuple[NDArray, NDArray] : (updated distribution, concentration)
        """
        if distribution.ndim == 2:
            concentration_expand = concentration[:, np.newaxis]
        else:
            concentration_expand = concentration

        new_mass = np.divide(
            np.maximum(distribution * concentration_expand + added_mass, 0),
            concentration_expand,
            out=np.zeros_like(distribution),
            where=concentration_expand != 0,
        )
        if new_mass.ndim == 1:
            new_mass_sum = np.sum(new_mass)
        else:
            new_mass_sum = np.sum(new_mass, axis=1)
        concentration = np.where(new_mass_sum > 0, concentration, 0)
        return (new_mass, concentration)

    def add_concentration(self, distribution, concentration,
                          added_distribution, added_concentration):
        """
        Add concentration to the distribution.

        Arguments:
            - distribution : NDArray[np.float64]
                Mass distribution array [kg].
            - concentration : NDArray[np.float64]
                Concentration array [1/m³].
            - added_distribution : NDArray[np.float64]
                Added mass distribution [kg].
            - added_concentration : NDArray[np.float64]
                Added concentration [1/m³].

        Returns:
            - Tuple[NDArray, NDArray] : (distribution, updated concentration)
        """
        rescaled = False
        if np.all(added_concentration == 1):
            rescaled = True
        if np.allclose(
            added_concentration, np.max(concentration), atol=1e-2
        ) or np.all(concentration == 0):
            added_concentration = added_concentration / np.max(concentration)
            rescaled = True
        if not rescaled:
            message = (
                "When adding concentration to ParticleResolvedSpeciatedMass, "
                "the added concentration should be all ones or all the same "
                "value of 1/volume."
            )
            raise ValueError(message)

        concentration = np.divide(
            concentration,
            concentration,
            out=np.zeros_like(concentration),
            where=concentration != 0,
        )

        empty_bins = np.flatnonzero(np.all(concentration == 0))
        empty_bins_count = len(empty_bins)
        added_bins_count = len(added_concentration)
        if empty_bins_count >= added_bins_count:
            distribution[empty_bins] = added_distribution
            concentration[empty_bins] = added_concentration
            return distribution, concentration
        if empty_bins_count > 0:
            distribution[empty_bins] = added_distribution[:empty_bins_count]
            concentration[empty_bins] = added_concentration[:empty_bins_count]
        distribution = np.concatenate(
            (distribution, added_distribution[empty_bins_count:]), axis=0
        )
        concentration = np.concatenate(
            (concentration, added_concentration[empty_bins_count:]), axis=0
        )
        return distribution, concentration

    def collide_pairs(self, distribution, concentration, density, indices):
        """
        Merge pairs of particles/bins by summing their distributions.

        Arguments:
            - distribution : NDArray[np.float64]
                Mass distribution array [kg].
            - concentration : NDArray[np.float64]
                Concentration array [1/m³].
            - density : NDArray[np.float64]
                Density array [kg/m³].
            - indices : NDArray
                Array of index pairs to merge.

        Returns:
            - Tuple[NDArray, NDArray] : (updated distribution, concentration)
        """
        small_index = indices[:, 0]
        large_index = indices[:, 1]
        if distribution.ndim == 1:
            distribution[large_index] += distribution[small_index]
            distribution[small_index] = 0
            concentration[small_index] = 0
            return distribution, concentration
        distribution[large_index, :] += distribution[small_index, :]
        distribution[small_index, :] = 0
        concentration[small_index] = 0
        return distribution, concentration
