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

ti.init(arch=ti.cpu, default_fp=ti.f64)  # guard against forgotten init


# ── generic Taichi helpers (no NumPy) ────────────────────────────────
@ti.kernel
def _ksum_axis1(
    mat: ti.types.ndarray(dtype=ti.f64, ndim=2),
    out_: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(mat.shape[0]):
        acc = 0.0
        for j in range(mat.shape[1]):
            acc += mat[i, j]
        out_[i] = acc


@ti.kernel
def _kdot(
    a: ti.types.ndarray(dtype=ti.f64, ndim=1),
    b: ti.types.ndarray(dtype=ti.f64, ndim=1),
) -> ti.f64:
    tot = 0.0
    for i in range(a.shape[0]):
        tot += a[i] * b[i]
    return tot


@ti.kernel
def _kadd(
    dst: ti.types.ndarray(dtype=ti.f64),
    src1: ti.types.ndarray(dtype=ti.f64),
    src2: ti.types.ndarray(dtype=ti.f64),
):
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
        return 4.0 / 3.0 * ti.math.pi * radius**3


# ──────────────────────────────────────────────────────────────────────
@ti.data_oriented
class TiParticleResolvedSpeciatedMass(_DistributionMixin):
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

    def get_name(self) -> str:  # parity with python back-end
        return self.__class__.__name__

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
        return distribution

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
        def _k_radius(
            d: ti.types.ndarray(dtype=ti.f64, ndim=2),
            rho: ti.types.ndarray(dtype=ti.f64, ndim=1),
            out_: ti.types.ndarray(dtype=ti.f64, ndim=1),
        ):
            for i in range(d.shape[0]):
                acc = 0.0
                for j in range(d.shape[1]):
                    acc += d[i, j] / rho[j]
                out_[i] = ti.pow(3.0 * acc / (4.0 * ti.math.pi), 1.0 / 3.0)

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

            for i, j in ti.ndrange(
                distribution.shape[0], distribution.shape[1]
            ):
                c = concentration[i]
                m_add = added_mass[i, j] / c if c > 0 else 0.0
                out_[i, j] = ti.max(distribution[i, j] + m_add, 0.0)


        result = ti.ndarray(dtype=ti.f64, shape=distribution.shape)
        _k_add_mass(distribution, concentration, added_mass, result)
        return result, concentration

    def add_concentration(
        self,
        distribution,
        concentration,
        added_distribution,
        added_concentration,
    ):
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
        def _k_collide_pairs_2d(
            distribution: ti.types.ndarray(dtype=ti.f64, ndim=2),
            concentration: ti.types.ndarray(dtype=ti.f64, ndim=1),
            indices: ti.types.ndarray(dtype=ti.f64, ndim=2),
        ):
            for k in range(indices.shape[0]):
                small_index = int(indices[k, 0])
                large_index = int(indices[k, 1])
                for j in range(distribution.shape[1]):
                    distribution[large_index, j] += distribution[
                        small_index, j
                    ]
                    distribution[small_index, j] = 0.0
                concentration[small_index] = 0.0

        _k_collide_pairs_2d(distribution, concentration, indices)
        return distribution, concentration
