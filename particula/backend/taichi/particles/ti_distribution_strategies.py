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
from particula.backend.dispatch_register import register

ti.init(arch=ti.cpu, default_fp=ti.f64)     # guard against forgotten init

# ── generic Taichi helpers (no NumPy) ────────────────────────────────
@ti.kernel
def _ksum_axis1(mat: ti.types.ndarray(dtype=ti.f64, ndim=2),
                out_: ti.types.ndarray(dtype=ti.f64, ndim=1)):
    for i in range(mat.shape[0]):
        acc = 0.0
        for j in range(mat.shape[1]):
            acc += mat[i, j]
        out_[i] = acc

@ti.kernel
def _kdot(a: ti.types.ndarray(dtype=ti.f64, ndim=1),
          b: ti.types.ndarray(dtype=ti.f64, ndim=1)) -> ti.f64:
    tot = 0.0
    for i in range(a.shape[0]):
        tot += a[i] * b[i]
    return tot

@ti.kernel
def _kadd(dst: ti.types.ndarray(dtype=ti.f64),
          src1: ti.types.ndarray(dtype=ti.f64),
          src2: ti.types.ndarray(dtype=ti.f64)):
    for I in ti.grouped(dst):
        dst[I] = src1[I] + src2[I]

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
        distribution: ti.types.ndarray(dtype=ti.f64),
        density:       ti.types.ndarray(dtype=ti.f64),
    ) -> ti.types.ndarray(dtype=ti.f64):
        """
        Compute the species-level mass for each bin/particle.

        All inputs and outputs are ti.types.ndarray(dtype=ti.f64).
        No data leave Taichi until explicitly requested by the user.

        Arguments:
            - distribution : ti.types.ndarray(dtype=ti.f64)
                Mass distribution array [kg].
            - density : ti.types.ndarray(dtype=ti.f64)
                Density array [kg/m³].

        Returns:
            - ti.types.ndarray(dtype=ti.f64) : Species mass array [kg].
        """
        result = ti.ndarray(dtype=ti.f64, shape=distribution.shape)
        if distribution.ndim == 1:
            self._kget_species_mass_1d(distribution, density, result)
        elif distribution.ndim == 2:
            self._kget_species_mass_2d(distribution, density, result)
        else:
            raise ValueError("Only 1-D or 2-D distributions are supported")
        return result

    def get_mass(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64),
        density:       ti.types.ndarray(dtype=ti.f64),
    ) -> ti.types.ndarray(dtype=ti.f64):
        """
        Compute the total mass for each bin/particle.

        Arguments:
            - distribution : ti.types.ndarray(dtype=ti.f64)
                Mass distribution array [kg].
            - density : ti.types.ndarray(dtype=ti.f64)
                Density array [kg/m³].

        Returns:
            - ti.types.ndarray(dtype=ti.f64) : Total mass per bin/particle [kg].
        """
        distribution = _to_numpy(distribution)
        density      = _to_numpy(density)
        species_mass = self.get_species_mass(distribution, density)
        if distribution.ndim == 1:
            return species_mass
        result = ti.ndarray(dtype=ti.f64, shape=(distribution.shape[0],))
        _ksum_axis1(species_mass, result)
        return result

    def get_total_mass(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64),
        concentration: ti.types.ndarray(dtype=ti.f64),
        density:       ti.types.ndarray(dtype=ti.f64),
    ) -> ti.f64:
        """
        Compute the total system mass.

        Arguments:
            - distribution : ti.types.ndarray(dtype=ti.f64)
                Mass distribution array [kg].
            - concentration : ti.types.ndarray(dtype=ti.f64)
                Concentration array [1/m³].
            - density : ti.types.ndarray(dtype=ti.f64)
                Density array [kg/m³].

        Returns:
            - ti.f64 : Total mass [kg].
        """
        particle_mass = self.get_mass(distribution, density)
        return float(_kdot(particle_mass, concentration))

    def get_radius(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64),
        density:       ti.types.ndarray(dtype=ti.f64),
    ) -> ti.types.ndarray(dtype=ti.f64):
        """
        Compute the radius for each bin/particle.

        Arguments:
            - distribution : ti.types.ndarray(dtype=ti.f64)
                Mass distribution array [kg].
            - density : ti.types.ndarray(dtype=ti.f64)
                Density array [kg/m³].

        Returns:
            - ti.types.ndarray(dtype=ti.f64) : Radii [m].
        """
        @ti.kernel
        def _k_radius_1d(self, m: ti.types.ndarray(dtype=ti.f64, ndim=1),
                               rho: ti.types.ndarray(dtype=ti.f64, ndim=1),
                               out_: ti.types.ndarray(dtype=ti.f64, ndim=1)):
            for i in range(m.shape[0]):
                v = m[i] / rho[i]
                out_[i] = ti.pow(3.0 * v / (4.0 * ti.math.pi), 1.0/3.0)

        @ti.kernel
        def _k_radius_2d(self, m: ti.types.ndarray(dtype=ti.f64, ndim=2),
                               rho: ti.types.ndarray(dtype=ti.f64, ndim=2),
                               out_: ti.types.ndarray(dtype=ti.f64, ndim=2)):
            for i, j in ti.ndrange(m.shape[0], m.shape[1]):
                v = m[i, j] / rho[i, j]
                out_[i, j] = ti.pow(3.0 * v / (4.0 * ti.math.pi), 1.0/3.0)

        result = ti.ndarray(dtype=ti.f64, shape=distribution.shape)
        if distribution.ndim == 1:
            self._k_radius_1d(distribution, density, result)
        else:
            self._k_radius_2d(distribution, density, result)
        return result

    def add_mass(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64),
        concentration: ti.types.ndarray(dtype=ti.f64),
        density:       ti.types.ndarray(dtype=ti.f64),
        added_mass:    ti.types.ndarray(dtype=ti.f64),
    ) -> tuple[ti.types.ndarray(dtype=ti.f64), ti.types.ndarray(dtype=ti.f64)]:
        """
        Add mass to the distribution.

        Arguments:
            - distribution : ti.types.ndarray(dtype=ti.f64)
                Mass distribution array [kg].
            - concentration : ti.types.ndarray(dtype=ti.f64)
                Concentration array [1/m³].
            - density : ti.types.ndarray(dtype=ti.f64)
                Density array [kg/m³].
            - added_mass : ti.types.ndarray(dtype=ti.f64)
                Mass to add [kg].

        Returns:
            - Tuple[ti.types.ndarray(dtype=ti.f64), ti.types.ndarray(dtype=ti.f64)] : (updated distribution, concentration)
        """
        result = ti.ndarray(dtype=ti.f64, shape=distribution.shape)
        _kadd(result, distribution, added_mass)
        return result, concentration

    def add_concentration(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64),
        concentration: ti.types.ndarray(dtype=ti.f64),
        added_distribution: ti.types.ndarray(dtype=ti.f64),
        added_concentration: ti.types.ndarray(dtype=ti.f64),
    ) -> tuple[ti.types.ndarray(dtype=ti.f64), ti.types.ndarray(dtype=ti.f64)]:
        """
        Add concentration to the distribution.

        Arguments:
            - distribution : ti.types.ndarray(dtype=ti.f64)
                Mass distribution array [kg].
            - concentration : ti.types.ndarray(dtype=ti.f64)
                Concentration array [1/m³].
            - added_distribution : ti.types.ndarray(dtype=ti.f64)
                Added mass distribution [kg].
            - added_concentration : ti.types.ndarray(dtype=ti.f64)
                Added concentration [1/m³].

        Returns:
            - Tuple[ti.types.ndarray(dtype=ti.f64), ti.types.ndarray(dtype=ti.f64)] : (distribution, updated concentration)
        """
        new_c = ti.ndarray(dtype=ti.f64, shape=concentration.shape)
        _kadd(new_c, concentration, added_concentration)
        return (distribution, new_c)

    def collide_pairs(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64),
        concentration: ti.types.ndarray(dtype=ti.f64),
        density:       ti.types.ndarray(dtype=ti.f64),
        indices:       ti.types.ndarray(dtype=ti.f64),
    ):
        """
        Colliding pairs in MassBasedMovingBin is not physically meaningful.

        Arguments:
            - distribution : ti.types.ndarray(dtype=ti.f64)
            - concentration : ti.types.ndarray(dtype=ti.f64)
            - density : ti.types.ndarray(dtype=ti.f64)
            - indices : ti.types.ndarray(dtype=ti.f64)

        Raises:
            - NotImplementedError
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
        distribution: ti.types.ndarray(dtype=ti.f64),
        density: ti.types.ndarray(dtype=ti.f64),
    ) -> ti.types.ndarray(dtype=ti.f64):
        """
        Compute the species-level mass for each bin/particle.

        All inputs and outputs are ti.types.ndarray(dtype=ti.f64).
        No data leave Taichi until explicitly requested by the user.

        Arguments:
            - distribution : ti.types.ndarray(dtype=ti.f64)
                Mass distribution array [kg].
            - density : ti.types.ndarray(dtype=ti.f64)
                Density array [kg/m³].

        Returns:
            - ti.types.ndarray(dtype=ti.f64) : Species mass array [kg].
        """
        result = ti.ndarray(dtype=ti.f64, shape=distribution.shape)
        if distribution.ndim == 1:
            self._kget_species_mass_1d(distribution, density, result)
        elif distribution.ndim == 2:
            self._kget_species_mass_2d(distribution, density, result)
        else:
            raise ValueError("Only 1-D or 2-D distributions are supported")
        return result

    def get_mass(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64),
        density: ti.types.ndarray(dtype=ti.f64),
    ) -> ti.types.ndarray(dtype=ti.f64):
        """
        Compute the total mass for each bin/particle.

        Arguments:
            - distribution : ti.types.ndarray(dtype=ti.f64)
                Mass distribution array [kg].
            - density : ti.types.ndarray(dtype=ti.f64)
                Density array [kg/m³].

        Returns:
            - ti.types.ndarray(dtype=ti.f64) : Total mass per bin/particle [kg].
        """
        distribution = _to_numpy(distribution)
        density      = _to_numpy(density)
        species_mass = self.get_species_mass(distribution, density)
        if distribution.ndim == 1:
            return species_mass
        result = ti.ndarray(dtype=ti.f64, shape=(distribution.shape[0],))
        _ksum_axis1(species_mass, result)
        return result

    def get_total_mass(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64),
        concentration: ti.types.ndarray(dtype=ti.f64),
        density: ti.types.ndarray(dtype=ti.f64),
    ) -> ti.f64:
        """
        Compute the total system mass.

        Arguments:
            - distribution : ti.types.ndarray(dtype=ti.f64)
                Mass distribution array [kg].
            - concentration : ti.types.ndarray(dtype=ti.f64)
                Concentration array [1/m³].
            - density : ti.types.ndarray(dtype=ti.f64)
                Density array [kg/m³].

        Returns:
            - ti.f64 : Total mass [kg].
        """
        particle_mass = self.get_mass(distribution, density)
        return float(_kdot(particle_mass, concentration))

    def get_radius(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64),
        density: ti.types.ndarray(dtype=ti.f64),
    ) -> ti.types.ndarray(dtype=ti.f64):
        """
        Return the radii (identity for this strategy).

        Arguments:
            - distribution : ti.types.ndarray(dtype=ti.f64)
                Radii array [m].
            - density : ti.types.ndarray(dtype=ti.f64)
                Density array [kg/m³].

        Returns:
            - ti.types.ndarray(dtype=ti.f64) : Radii [m].
        """
        return distribution

    def add_mass(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64),
        concentration: ti.types.ndarray(dtype=ti.f64),
        density: ti.types.ndarray(dtype=ti.f64),
        added_mass: ti.types.ndarray(dtype=ti.f64),
    ) -> tuple[ti.types.ndarray(dtype=ti.f64), ti.types.ndarray(dtype=ti.f64)]:
        """
        Add mass to the distribution.

        Arguments:
            - distribution : ti.types.ndarray(dtype=ti.f64)
                Radii array [m].
            - concentration : ti.types.ndarray(dtype=ti.f64)
                Concentration array [1/m³].
            - density : ti.types.ndarray(dtype=ti.f64)
                Density array [kg/m³].
            - added_mass : ti.types.ndarray(dtype=ti.f64)
                Mass to add [kg].

        Returns:
            - Tuple[ti.types.ndarray(dtype=ti.f64), ti.types.ndarray(dtype=ti.f64)] : (updated radii, concentration)
        """
        @ti.kernel
        def _k_add_mass(
            distribution: ti.types.ndarray(dtype=ti.f64),
            concentration: ti.types.ndarray(dtype=ti.f64),
            density: ti.types.ndarray(dtype=ti.f64),
            added_mass: ti.types.ndarray(dtype=ti.f64),
            out_: ti.types.ndarray(dtype=ti.f64),
        ):
            for I in ti.grouped(distribution):
                c = concentration[I]
                m_add = added_mass[I] / c if c > 0 else 0.0
                v0 = (4.0 / 3.0) * ti.math.pi * distribution[I] ** 3
                v1 = v0 + m_add / density[I]
                out_[I] = ti.pow(3.0 * v1 / (4.0 * ti.math.pi), 1.0 / 3.0)
        result = ti.ndarray(dtype=ti.f64, shape=distribution.shape)
        self._k_add_mass(distribution, concentration, density, added_mass, result)
        return result, concentration

    def add_concentration(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64),
        concentration: ti.types.ndarray(dtype=ti.f64),
        added_distribution: ti.types.ndarray(dtype=ti.f64),
        added_concentration: ti.types.ndarray(dtype=ti.f64),
    ) -> tuple[ti.types.ndarray(dtype=ti.f64), ti.types.ndarray(dtype=ti.f64)]:
        """
        Add concentration to the distribution.

        Arguments:
            - distribution : ti.types.ndarray(dtype=ti.f64)
                Radii array [m].
            - concentration : ti.types.ndarray(dtype=ti.f64)
                Concentration array [1/m³].
            - added_distribution : ti.types.ndarray(dtype=ti.f64)
                Added radii [m].
            - added_concentration : ti.types.ndarray(dtype=ti.f64)
                Added concentration [1/m³].

        Returns:
            - Tuple[ti.types.ndarray(dtype=ti.f64), ti.types.ndarray(dtype=ti.f64)] : (distribution, updated concentration)
        """
        new_c = ti.ndarray(dtype=ti.f64, shape=concentration.shape)
        _kadd(new_c, concentration, added_concentration)
        return (distribution, new_c)

    def collide_pairs(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64),
        concentration: ti.types.ndarray(dtype=ti.f64),
        density: ti.types.ndarray(dtype=ti.f64),
        indices: ti.types.ndarray(dtype=ti.f64),
    ):
        """
        Colliding pairs in RadiiBasedMovingBin is not physically meaningful.

        Arguments:
            - distribution : ti.types.ndarray(dtype=ti.f64)
            - concentration : ti.types.ndarray(dtype=ti.f64)
            - density : ti.types.ndarray(dtype=ti.f64)
            - indices : ti.types.ndarray(dtype=ti.f64)

        Raises:
            - NotImplementedError
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
        distribution: ti.types.ndarray(dtype=ti.f64),
        density: ti.types.ndarray(dtype=ti.f64),
    ) -> ti.types.ndarray(dtype=ti.f64):
        """
        Compute the species-level mass for each bin/particle.

        All inputs and outputs are ti.types.ndarray(dtype=ti.f64).
        No data leave Taichi until explicitly requested by the user.

        Arguments:
            - distribution : ti.types.ndarray(dtype=ti.f64)
                Mass distribution array [kg].
            - density : ti.types.ndarray(dtype=ti.f64)
                Density array [kg/m³].

        Returns:
            - ti.types.ndarray(dtype=ti.f64) : Species mass array [kg].
        """
        result = ti.ndarray(dtype=ti.f64, shape=distribution.shape)
        if distribution.ndim == 1:
            self._kget_species_mass_1d(distribution, density, result)
        elif distribution.ndim == 2:
            self._kget_species_mass_2d(distribution, density, result)
        else:
            raise ValueError("Only 1-D or 2-D distributions are supported")
        return result

    def get_mass(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64),
        density: ti.types.ndarray(dtype=ti.f64),
    ) -> ti.types.ndarray(dtype=ti.f64):
        """
        Compute the total mass for each bin/particle.

        Arguments:
            - distribution : ti.types.ndarray(dtype=ti.f64)
                Mass distribution array [kg].
            - density : ti.types.ndarray(dtype=ti.f64)
                Density array [kg/m³].

        Returns:
            - ti.types.ndarray(dtype=ti.f64) : Total mass per bin/particle [kg].
        """
        distribution = _to_numpy(distribution)
        density      = _to_numpy(density)
        species_mass = self.get_species_mass(distribution, density)
        if distribution.ndim == 1:
            return species_mass
        result = ti.ndarray(dtype=ti.f64, shape=(distribution.shape[0],))
        _ksum_axis1(species_mass, result)
        return result

    def get_total_mass(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64),
        concentration: ti.types.ndarray(dtype=ti.f64),
        density: ti.types.ndarray(dtype=ti.f64),
    ) -> ti.f64:
        """
        Compute the total system mass.

        Arguments:
            - distribution : ti.types.ndarray(dtype=ti.f64)
                Mass distribution array [kg].
            - concentration : ti.types.ndarray(dtype=ti.f64)
                Concentration array [1/m³].
            - density : ti.types.ndarray(dtype=ti.f64)
                Density array [kg/m³].

        Returns:
            - ti.f64 : Total mass [kg].
        """
        particle_mass = self.get_mass(distribution, density)
        return float(_kdot(particle_mass, concentration))

    def get_radius(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64),
        density: ti.types.ndarray(dtype=ti.f64),
    ) -> ti.types.ndarray(dtype=ti.f64):
        """
        Compute the radius for each bin/particle.

        Arguments:
            - distribution : ti.types.ndarray(dtype=ti.f64)
                Mass distribution array [kg].
            - density : ti.types.ndarray(dtype=ti.f64)
                Density array [kg/m³].

        Returns:
            - ti.types.ndarray(dtype=ti.f64) : Radii [m].
        """
        @ti.kernel
        def _k_radius(self, d: ti.types.ndarray(dtype=ti.f64, ndim=2),
                            rho: ti.types.ndarray(dtype=ti.f64, ndim=2),
                            out_: ti.types.ndarray(dtype=ti.f64, ndim=1)):
            for i in range(d.shape[0]):
                acc = 0.0
                for j in range(d.shape[1]):
                    acc += d[i, j] / rho[i, j]
                out_[i] = ti.pow(3.0 * acc / (4.0 * ti.math.pi), 1.0/3.0)
        if distribution.ndim == 1:
            result = ti.ndarray(dtype=ti.f64, shape=distribution.shape)
            for i in range(result.shape[0]):
                result[i] = ti.pow(3.0 * (distribution[i] / density[i]) / (4.0 * ti.math.pi), 1.0/3.0)
            return result
        else:
            result = ti.ndarray(dtype=ti.f64, shape=(distribution.shape[0],))
            self._k_radius(distribution, density, result)
            return result

    def add_mass(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64),
        concentration: ti.types.ndarray(dtype=ti.f64),
        density: ti.types.ndarray(dtype=ti.f64),
        added_mass: ti.types.ndarray(dtype=ti.f64),
    ) -> tuple[ti.types.ndarray(dtype=ti.f64), ti.types.ndarray(dtype=ti.f64)]:
        """
        Add mass to the distribution.

        Arguments:
            - distribution : ti.types.ndarray(dtype=ti.f64)
                Mass distribution array [kg].
            - concentration : ti.types.ndarray(dtype=ti.f64)
                Concentration array [1/m³].
            - density : ti.types.ndarray(dtype=ti.f64)
                Density array [kg/m³].
            - added_mass : ti.types.ndarray(dtype=ti.f64)
                Mass to add [kg].

        Returns:
            - Tuple[ti.types.ndarray(dtype=ti.f64), ti.types.ndarray(dtype=ti.f64)] : (updated distribution, concentration)
        """
        @ti.kernel
        def _k_add_mass(
            distribution: ti.types.ndarray(dtype=ti.f64),
            concentration: ti.types.ndarray(dtype=ti.f64),
            added_mass: ti.types.ndarray(dtype=ti.f64),
            out_: ti.types.ndarray(dtype=ti.f64),
        ):
            if distribution.ndim == 2:
                for i, j in ti.ndrange(distribution.shape[0], distribution.shape[1]):
                    c = concentration[i]
                    m_add = added_mass[i, j] / c if c > 0 else 0.0
                    out_[i, j] = ti.max(distribution[i, j] + m_add, 0.0)
            else:
                for i in range(distribution.shape[0]):
                    c = concentration[i]
                    m_add = added_mass[i] / c if c > 0 else 0.0
                    out_[i] = ti.max(distribution[i] + m_add, 0.0)
        result = ti.ndarray(dtype=ti.f64, shape=distribution.shape)
        _k_add_mass(distribution, concentration, added_mass, result)
        return result, concentration

    def add_concentration(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64),
        concentration: ti.types.ndarray(dtype=ti.f64),
        added_distribution: ti.types.ndarray(dtype=ti.f64),
        added_concentration: ti.types.ndarray(dtype=ti.f64),
    ) -> tuple[ti.types.ndarray(dtype=ti.f64), ti.types.ndarray(dtype=ti.f64)]:
        """
        Add concentration to the distribution.

        Arguments:
            - distribution : ti.types.ndarray(dtype=ti.f64)
                Mass distribution array [kg].
            - concentration : ti.types.ndarray(dtype=ti.f64)
                Concentration array [1/m³].
            - added_distribution : ti.types.ndarray(dtype=ti.f64)
                Added mass distribution [kg].
            - added_concentration : ti.types.ndarray(dtype=ti.f64)
                Added concentration [1/m³].

        Returns:
            - Tuple[ti.types.ndarray(dtype=ti.f64), ti.types.ndarray(dtype=ti.f64)] : (distribution, updated concentration)
        """
        new_c = ti.ndarray(dtype=ti.f64, shape=concentration.shape)
        _kadd(new_c, concentration, added_concentration)
        return (distribution, new_c)

    def collide_pairs(
        self,
        distribution: ti.types.ndarray(dtype=ti.f64),
        concentration: ti.types.ndarray(dtype=ti.f64),
        density: ti.types.ndarray(dtype=ti.f64),
        indices: ti.types.ndarray(dtype=ti.f64),
    ):
        """
        Colliding pairs in SpeciatedMassMovingBin is not physically meaningful.

        Arguments:
            - distribution : ti.types.ndarray(dtype=ti.f64)
            - concentration : ti.types.ndarray(dtype=ti.f64)
            - density : ti.types.ndarray(dtype=ti.f64)
            - indices : ti.types.ndarray(dtype=ti.f64)

        Raises:
            - NotImplementedError
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
        distribution: ti.types.ndarray(dtype=ti.f64),
        density: ti.types.ndarray(dtype=ti.f64),
    ) -> ti.types.ndarray(dtype=ti.f64):
        """
        Compute the species-level mass for each bin/particle.

        All inputs and outputs are ti.types.ndarray(dtype=ti.f64).
        No data leave Taichi until explicitly requested by the user.

        Arguments:
            - distribution : ti.types.ndarray(dtype=ti.f64)
                Mass distribution array [kg].
            - density : ti.types.ndarray(dtype=ti.f64)
                Density array [kg/m³].

        Returns:
            - ti.types.ndarray(dtype=ti.f64) : Species mass array [kg].
        """
        result = ti.ndarray(dtype=ti.f64, shape=distribution.shape)
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

        All inputs and outputs are ti.types.ndarray(dtype=ti.f64).
        No data leave Taichi until explicitly requested by the user.

        Arguments:
            - distribution : ti.types.ndarray(dtype=ti.f64)
                Mass distribution array [kg].
            - density : ti.types.ndarray(dtype=ti.f64)
                Density array [kg/m³].

        Returns:
            - ti.types.ndarray(dtype=ti.f64) : Total mass per bin/particle [kg].
        """
        species_mass = self.get_species_mass(distribution, density)
        if distribution.ndim == 1:
            return species_mass
        result = ti.ndarray(dtype=ti.f64, shape=(distribution.shape[0],))
        _ksum_axis1(species_mass, result)
        return result

    def get_total_mass(self, distribution, concentration, density):
        """
        Compute the total system mass.

        All inputs and outputs are ti.types.ndarray(dtype=ti.f64).
        No data leave Taichi until explicitly requested by the user.

        Arguments:
            - distribution : ti.types.ndarray(dtype=ti.f64)
                Mass distribution array [kg].
            - concentration : ti.types.ndarray(dtype=ti.f64)
                Concentration array [1/m³].
            - density : ti.types.ndarray(dtype=ti.f64)
                Density array [kg/m³].

        Returns:
            - float : Total mass [kg].
        """
        particle_mass = self.get_mass(distribution, density)
        return float(_kdot(particle_mass, concentration))

    def get_radius(self, distribution, density):
        """
        Compute the radius for each bin/particle.

        All inputs and outputs are ti.types.ndarray(dtype=ti.f64).
        No data leave Taichi until explicitly requested by the user.

        Arguments:
            - distribution : ti.types.ndarray(dtype=ti.f64)
                Mass distribution array [kg].
            - density : ti.types.ndarray(dtype=ti.f64)
                Density array [kg/m³].

        Returns:
            - ti.types.ndarray(dtype=ti.f64) : Radii [m].
        """
        @ti.kernel
        def _k_radius(d: ti.types.ndarray(dtype=ti.f64, ndim=2),
                      rho: ti.types.ndarray(dtype=ti.f64, ndim=2),
                      out_: ti.types.ndarray(dtype=ti.f64, ndim=1)):
            for i in range(d.shape[0]):
                acc = 0.0
                for j in range(d.shape[1]):
                    acc += d[i, j] / rho[i, j]
                out_[i] = ti.pow(3.0 * acc / (4.0 * ti.math.pi), 1.0/3.0)
        if distribution.ndim == 1:
            result = ti.ndarray(dtype=ti.f64, shape=distribution.shape)
            for i in range(result.shape[0]):
                result[i] = ti.pow(3.0 * (distribution[i] / density[i]) / (4.0 * ti.math.pi), 1.0/3.0)
            return result
        else:
            result = ti.ndarray(dtype=ti.f64, shape=(distribution.shape[0],))
            _k_radius(distribution, density, result)
            return result

    def add_mass(self, distribution, concentration, density, added_mass):
        """
        Add mass to the distribution.

        All inputs and outputs are ti.types.ndarray(dtype=ti.f64).
        No data leave Taichi until explicitly requested by the user.

        Arguments:
            - distribution : ti.types.ndarray(dtype=ti.f64)
                Mass distribution array [kg].
            - concentration : ti.types.ndarray(dtype=ti.f64)
                Concentration array [1/m³].
            - density : ti.types.ndarray(dtype=ti.f64)
                Density array [kg/m³].
            - added_mass : ti.types.ndarray(dtype=ti.f64)
                Mass to add [kg].

        Returns:
            - Tuple[ti.types.ndarray(dtype=ti.f64), ti.types.ndarray(dtype=ti.f64)] : (updated distribution, concentration)
        """
        @ti.kernel
        def _k_add_mass(
            distribution: ti.types.ndarray(dtype=ti.f64),
            concentration: ti.types.ndarray(dtype=ti.f64),
            added_mass: ti.types.ndarray(dtype=ti.f64),
            out_: ti.types.ndarray(dtype=ti.f64),
        ):
            if distribution.ndim == 2:
                for i, j in ti.ndrange(distribution.shape[0], distribution.shape[1]):
                    c = concentration[i]
                    m_add = added_mass[i, j] / c if c > 0 else 0.0
                    out_[i, j] = ti.max(distribution[i, j] + m_add, 0.0)
            else:
                for i in range(distribution.shape[0]):
                    c = concentration[i]
                    m_add = added_mass[i] / c if c > 0 else 0.0
                    out_[i] = ti.max(distribution[i] + m_add, 0.0)
        result = ti.ndarray(dtype=ti.f64, shape=distribution.shape)
        _k_add_mass(distribution, concentration, added_mass, result)
        return result, concentration

    def add_concentration(self, distribution, concentration,
                          added_distribution, added_concentration):
        """
        Add concentration to the distribution.

        All inputs and outputs are ti.types.ndarray(dtype=ti.f64).
        No data leave Taichi until explicitly requested by the user.

        Arguments:
            - distribution : ti.types.ndarray(dtype=ti.f64)
                Mass distribution array [kg].
            - concentration : ti.types.ndarray(dtype=ti.f64)
                Concentration array [1/m³].
            - added_distribution : ti.types.ndarray(dtype=ti.f64)
                Added mass distribution [kg].
            - added_concentration : ti.types.ndarray(dtype=ti.f64)
                Added concentration [1/m³].

        Returns:
            - Tuple[ti.types.ndarray(dtype=ti.f64), ti.types.ndarray(dtype=ti.f64)] : (distribution, updated concentration)
        """
        new_c = ti.ndarray(dtype=ti.f64, shape=concentration.shape)
        _kadd(new_c, concentration, added_concentration)
        return (distribution, new_c)

    def collide_pairs(self, distribution, concentration, density, indices):
        """
        Merge pairs of particles/bins by summing their distributions.

        All inputs and outputs are ti.types.ndarray(dtype=ti.f64).
        No data leave Taichi until explicitly requested by the user.

        Arguments:
            - distribution : ti.types.ndarray(dtype=ti.f64)
                Mass distribution array [kg].
            - concentration : ti.types.ndarray(dtype=ti.f64)
                Concentration array [1/m³].
            - density : ti.types.ndarray(dtype=ti.f64)
                Density array [kg/m³].
            - indices : ti.types.ndarray(dtype=ti.f64)
                Array of index pairs to merge.

        Returns:
            - Tuple[ti.types.ndarray(dtype=ti.f64), ti.types.ndarray(dtype=ti.f64)] : (updated distribution, concentration)
        """
        @ti.kernel
        def _k_collide_pairs_1d(
            distribution: ti.types.ndarray(dtype=ti.f64, ndim=1),
            concentration: ti.types.ndarray(dtype=ti.f64, ndim=1),
            indices: ti.types.ndarray(dtype=ti.f64, ndim=2),
        ):
            for k in range(indices.shape[0]):
                small_index = int(indices[k, 0])
                large_index = int(indices[k, 1])
                distribution[large_index] += distribution[small_index]
                distribution[small_index] = 0.0
                concentration[small_index] = 0.0

        @ti.kernel
        def _k_collide_pairs_2d(
            distribution: ti.types.ndarray(dtype=ti.f64, ndim=2),
            concentration: ti.types.ndarray(dtype=ti.f64, ndim=1),
            indices: ti.types.ndarray(dtype=ti.f64, ndim=2),
        ):
            for k in range(indices.shape[0]):
                small_index = int(indices[k, 0])
                large_index = int(indices[k, 1])
                for j in range(distribution.shape[1]):
                    distribution[large_index, j] += distribution[small_index, j]
                    distribution[small_index, j] = 0.0
                concentration[small_index] = 0.0

        if distribution.ndim == 1:
            _k_collide_pairs_1d(distribution, concentration, indices)
        else:
            _k_collide_pairs_2d(distribution, concentration, indices)
        return distribution, concentration
