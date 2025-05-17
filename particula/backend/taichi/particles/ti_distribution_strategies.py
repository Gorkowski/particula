"""Taichi implementation of particula.particles.distribution_strategies.*"""
import taichi as ti
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from particula.backend.dispatch_register import register

ti.init(arch=ti.cpu, default_fp=ti.f64)     # guard against forgotten init

# ─── mix-in holding helpers used in several subclasses ─────────────────
@ti.data_oriented
class _DistributionMixin:
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
    Taichi drop-in replacement for particula.particles.distribution_strategies.MassBasedMovingBin.
    Mirrors the public API (same method names & signatures).
    """
    def __init__(self):
        pass

    @ti.func
    def _get_species_mass_func(self, distribution, density):
        return distribution

    @ti.kernel
    def _get_species_mass_kernel(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64, ndim=ti.any_ndim),
        density:       ti.types.ndarray(dtype=ti.f64, ndim=ti.any_ndim),
        result:        ti.types.ndarray(dtype=ti.f64, ndim=ti.any_ndim),
    ):
        """Populate *result* with mass per species for every bin."""
        for I in ti.grouped(distribution):
            result[I] = self._get_species_mass_func(distribution[I], density[I])

    def get_species_mass(
        self,
        distribution: NDArray[np.float64],
        density: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        result = np.empty_like(distribution)
        self._get_species_mass_kernel(distribution, density, result)
        return result

    def get_mass(self, distribution, density):
        species_mass = self.get_species_mass(distribution, density)
        if distribution.ndim == 1:
            return species_mass
        return np.sum(species_mass, axis=1)

    def get_total_mass(self, distribution, concentration, density):
        return np.sum(self.get_mass(distribution, density) * concentration)

    def get_radius(self, distribution, density):
        # Calculate the volume of each particle from its mass and density,
        # then calculate the radius.
        volumes = distribution / density
        return (3 * volumes / (4 * np.pi)) ** (1 / 3)

    def add_mass(self, distribution, concentration, density, added_mass):
        # Add the mass to the distribution moving the bins
        return (distribution + added_mass, concentration)

    def add_concentration(self, distribution, concentration,
                          added_distribution, added_concentration):
        if (distribution.shape != added_distribution.shape) and (
            np.allclose(distribution, added_distribution, rtol=1e-6)
        ):
            message = (
                "When adding concentration to MassBasedMovingBin,"
                + " the distribution and added distribution should have "
                "the same elements. The current distribution shape is "
                f"{distribution.shape} and the added distribution shape is "
                f"{added_distribution.shape}."
            )
            import logging
            logger = logging.getLogger("particula")
            logger.error(message)
            raise ValueError(message)
        if concentration.shape != added_concentration.shape:
            message = (
                "When adding concentration to MassBasedMovingBin,"
                " the concentration and added concentration should have "
                "the same shape. The current concentration shape is "
                f"{concentration.shape} and the added concentration shape is "
                f"{added_concentration.shape}."
            )
            import logging
            logger = logging.getLogger("particula")
            logger.error(message)
            raise ValueError(message)
        concentration += added_concentration
        return (distribution, concentration)

    def collide_pairs(self, distribution, concentration, density, indices):
        message = (
            "Colliding pairs in MassBasedMovingBin is not physically"
            + "meaningful, change dyanmic or particle strategy."
        )
        import logging
        logger = logging.getLogger("particula")
        logger.warning(message)
        raise NotImplementedError(message)

# ──────────────────────────────────────────────────────────────────────
@register("RadiiBasedMovingBin", backend="taichi")
@ti.data_oriented
class RadiiBasedMovingBin(_DistributionMixin):
    """
    Taichi drop-in replacement for particula.particles.distribution_strategies.RadiiBasedMovingBin.
    Mirrors the public API (same method names & signatures).
    """
    def __init__(self):
        pass

    @ti.func
    def _get_species_mass_func(self, distribution, density):
        return 4.0 / 3.0 * ti.math.pi * distribution ** 3 * density

    @ti.kernel
    def _get_species_mass_kernel(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64, ndim=ti.any_ndim),
        density:       ti.types.ndarray(dtype=ti.f64, ndim=ti.any_ndim),
        result:        ti.types.ndarray(dtype=ti.f64, ndim=ti.any_ndim),
    ):
        """Populate *result* with mass per species for every bin."""
        for I in ti.grouped(distribution):
            result[I] = self._get_species_mass_func(distribution[I], density[I])

    def get_species_mass(
        self,
        distribution: NDArray[np.float64],
        density: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        result = np.empty_like(distribution)
        self._get_species_mass_kernel(distribution, density, result)
        return result

    def get_mass(self, distribution, density):
        species_mass = self.get_species_mass(distribution, density)
        if distribution.ndim == 1:
            return species_mass
        return np.sum(species_mass, axis=1)

    def get_total_mass(self, distribution, concentration, density):
        return np.sum(self.get_mass(distribution, density) * concentration)

    def get_radius(self, distribution, density):
        return distribution

    def add_mass(self, distribution, concentration, density, added_mass):
        mass_per_particle = np.where(
            concentration > 0, added_mass / concentration, 0
        )
        initial_volumes = (4 / 3) * np.pi * np.power(distribution, 3)
        new_volumes = initial_volumes + mass_per_particle / density
        new_radii = np.power(3 * new_volumes / (4 * np.pi), 1 / 3)
        return (new_radii, concentration)

    def add_concentration(self, distribution, concentration,
                          added_distribution, added_concentration):
        if (distribution.shape != added_distribution.shape) and (
            np.allclose(distribution, added_distribution, rtol=1e-6)
        ):
            message = (
                "When adding concentration to RadiiBasedMovingBin,"
                " the distribution and added distribution should have "
                "the same elements. The current distribution shape is "
                f"{distribution.shape} and the added distribution shape is "
                f"{added_distribution.shape}."
            )
            import logging
            logger = logging.getLogger("particula")
            logger.error(message)
            raise ValueError(message)
        if concentration.shape != added_concentration.shape:
            message = (
                "When adding concentration to RadiiBasedMovingBin,"
                " the concentration and added concentration should have "
                "the same shape. The current concentration shape is "
                f"{concentration.shape} and the added concentration shape is "
                f"{added_concentration.shape}."
            )
            import logging
            logger = logging.getLogger("particula")
            logger.error(message)
            raise ValueError(message)
        concentration += added_concentration
        return (distribution, concentration)

    def collide_pairs(self, distribution, concentration, density, indices):
        message = (
            "Colliding pairs in RadiiBasedMovingBin is not physically"
            + "meaningful, change dyanmic or particle strategy."
        )
        import logging
        logger = logging.getLogger("particula")
        logger.warning(message)
        raise NotImplementedError(message)

# ──────────────────────────────────────────────────────────────────────
@register("SpeciatedMassMovingBin", backend="taichi")
@ti.data_oriented
class SpeciatedMassMovingBin(_DistributionMixin):
    """
    Taichi drop-in replacement for particula.particles.distribution_strategies.SpeciatedMassMovingBin.
    Mirrors the public API (same method names & signatures).
    """
    def __init__(self):
        pass

    @ti.func
    def _get_species_mass_func(self, distribution, density):
        return distribution

    @ti.kernel
    def _get_species_mass_kernel(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64, ndim=ti.any_ndim),
        density:       ti.types.ndarray(dtype=ti.f64, ndim=ti.any_ndim),
        result:        ti.types.ndarray(dtype=ti.f64, ndim=ti.any_ndim),
    ):
        """Populate *result* with mass per species for every bin."""
        for I in ti.grouped(distribution):
            result[I] = self._get_species_mass_func(distribution[I], density[I])

    def get_species_mass(
        self,
        distribution: NDArray[np.float64],
        density: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        result = np.empty_like(distribution)
        self._get_species_mass_kernel(distribution, density, result)
        return result

    def get_mass(self, distribution, density):
        species_mass = self.get_species_mass(distribution, density)
        if distribution.ndim == 1:
            return species_mass
        return np.sum(species_mass, axis=1)

    def get_total_mass(self, distribution, concentration, density):
        return np.sum(self.get_mass(distribution, density) * concentration)

    def get_radius(self, distribution, density):
        volumes = np.sum(distribution / density, axis=1)
        return (3 * volumes / (4 * np.pi)) ** (1 / 3)

    def add_mass(self, distribution, concentration, density, added_mass):
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
        if (distribution.shape != added_distribution.shape) and (
            np.allclose(distribution, added_distribution, rtol=1e-6)
        ):
            message = (
                "When adding concentration to SpeciatedMassMovingBin,"
                " the distribution and added distribution should have "
                "the same elements. The current distribution shape is "
                f"{distribution.shape} and the added distribution shape is "
                f"{added_distribution.shape}."
            )
            import logging
            logger = logging.getLogger("particula")
            logger.error(message)
            raise ValueError(message)
        if concentration.shape != added_concentration.shape:
            message = (
                "When adding concentration to SpeciatedMassMovingBin,"
                " the concentration and added concentration should have "
                "the same shape. The current concentration shape is "
                f"{concentration.shape} and the added concentration shape is "
                f"{added_concentration.shape}."
            )
            import logging
            logger = logging.getLogger("particula")
            logger.error(message)
            raise ValueError(message)
        concentration += added_concentration
        return (distribution, concentration)

    def collide_pairs(self, distribution, concentration, density, indices):
        message = (
            "Colliding pairs in SpeciatedMassMovingBin is not physically"
            + "meaningful, change dyanmic or particle strategy."
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
    Taichi drop-in replacement for particula.particles.distribution_strategies.ParticleResolvedSpeciatedMass.
    Mirrors the public API (same method names & signatures).
    """
    def __init__(self):
        pass

    @ti.func
    def _get_species_mass_func(self, distribution, density):
        return distribution

    @ti.kernel
    def _get_species_mass_kernel(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64, ndim=ti.any_ndim),
        density:       ti.types.ndarray(dtype=ti.f64, ndim=ti.any_ndim),
        result:        ti.types.ndarray(dtype=ti.f64, ndim=ti.any_ndim),
    ):
        """Populate *result* with mass per species for every bin."""
        for I in ti.grouped(distribution):
            result[I] = self._get_species_mass_func(distribution[I], density[I])

    def get_species_mass(
        self,
        distribution: NDArray[np.float64],
        density: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        result = np.empty_like(distribution)
        self._get_species_mass_kernel(distribution, density, result)
        return result

    def get_mass(self, distribution, density):
        species_mass = self.get_species_mass(distribution, density)
        if distribution.ndim == 1:
            return species_mass
        return np.sum(species_mass, axis=1)

    def get_total_mass(self, distribution, concentration, density):
        return np.sum(self.get_mass(distribution, density) * concentration)

    def get_radius(self, distribution, density):
        if distribution.ndim == 1:
            volumes = distribution / density
        else:
            volumes = np.sum(distribution / density, axis=1)
        return (3 * volumes / (4 * np.pi)) ** (1 / 3)

    def add_mass(self, distribution, concentration, density, added_mass):
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
                "When adding concentration to ParticleResolvedSpeciatedMass,"
                + " the added concentration should be all ones or all the same"
                + " value of 1/volume."
            )
            import logging
            logger = logging.getLogger("particula")
            logger.error(message)
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
