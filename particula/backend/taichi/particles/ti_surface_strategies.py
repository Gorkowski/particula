"""
Taichi-accelerated surface property strategies for particula.

This module provides Taichi-accelerated drop-in replacements for
surface property mixing strategies used in particula's particle
microphysics models. It implements three strategies for computing
effective surface tension and density of multi-component droplets:
mole-fraction, mass-fraction, and volume-fraction weighted. These
are used to compute the Kelvin radius and Kelvin term, which are
critical for modeling vapor pressure and condensation/evaporation
in atmospheric aerosols.


References:
    - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric Chemistry
      and Physics: From Air Pollution to Climate Change (3rd ed.).
      Wiley.
    - "Kelvin equation," Wikipedia.
      https://en.wikipedia.org/wiki/Kelvin_equation

"""

import taichi as ti
from numpy.typing import NDArray
import numpy as np
from particula.backend.taichi.particles.properties.ti_kelvin_effect_module import (
    kget_kelvin_radius,
    kget_kelvin_term,
)

from particula.backend.taichi.util.ti_field_helper import FieldIO

_field_io = FieldIO()

@ti.data_oriented
class SurfaceMixin:
    """
    Shared helpers reused by the three pure-surface strategies.

    This mixin provides storage and access for surface tension and density
    properties, as well as Taichi-accelerated helpers for the Kelvin effect.

    Attributes:
        - surface_tension : Taichi field for surface tension.
        - density : Taichi field for density.
        - n_species : Number of species/components.

    Methods:
        - get_name: Returns the class name.

    """

    def __init__(
        self,
        surface_tension: NDArray[np.float64],
        density: NDArray[np.float64],
    ):
        self.surface_tension = ti.field(ti.f64, shape=surface_tension.shape)
        self.n_species = ti.field(ti.i32, shape=())
        self.density = ti.field(ti.f64, shape=density.shape)
        _field_io.from_numpy(self.surface_tension, surface_tension)
        _field_io.from_numpy(self.density, density)
        self.n_species[None] = int(surface_tension.shape[0])

    @ti.kernel
    def weighted_average(
        self,
        values: ti.template(),
        weights: ti.template(),
        normalizer: ti.template(),
    ) -> ti.f64:
        tot = 0.0
        for i in range(self.n_species):
            tot += weights[i] / normalizer[i]
        acc = 0.0
        for i in range(self.n_species):
            acc += values[i] * (weights[i] / normalizer[i]) / tot
        return acc

    def get_name(self) -> str:
        """
        Return the class name.

        Returns:
            - name : Name of the class as a string.

        Examples:
            >>> strat = SurfaceStrategyMolar()
            >>> strat.get_name()
            'SurfaceStrategyMolar'
        """
        return self.__class__.__name__


@ti.data_oriented
class TiSurfaceStrategyMolar(SurfaceMixin):
    """
    Surface property mixing by mole fraction (molar-based).

    Computes effective surface tension and density using mole-fraction
    weighting, suitable for ideal solutions. Provides Taichi-accelerated
    Kelvin radius and Kelvin term calculations.

    Attributes:
        - surface_tension : Taichi field for surface tension.
        - density : Taichi field for density.
        - molar_mass : Taichi field for molar mass.
        - n_species : Number of species/components.

    Methods:
        - effective_surface_tension: Mole-fraction weighted surface tension.
        - effective_density: Mole-fraction weighted density.
        - kelvin_radius: Kelvin radius (rₖ = (2σM)/(ρRT)).
        - kelvin_term: Kelvin term (exp(rₖ/r)).


    References:
        - Seinfeld & Pandis (2016), Atmospheric Chemistry and Physics.
        - "Kelvin equation," Wikipedia.
    """

    def __init__(
        self,
        surface_tension: ti.types.ndarray(dtype=ti.f64),
        density: ti.types.ndarray(dtype=ti.f64),
        molar_mass: ti.types.ndarray(dtype=ti.f64),
    ):
        super().__init__(surface_tension, density)
        self.molar_mass = molar_mass

    def effective_surface_tension(
        self, mass_concentration: ti.types.ndarray(dtype=ti.f64)
    ):
        return self.weighted_average(
            self.surface_tension, mass_concentration, self.molar_mass
        )

    def effective_density(
        self, mass_concentration: ti.types.ndarray(dtype=ti.f64)
    ):
        return self.weighted_average(
            self.density, mass_concentration, self.molar_mass
        )

    def kelvin_radius(
        self,
        molar_mass: ti.types.ndarray(dtype=ti.f64),
        mass_concentration: ti.types.ndarray(dtype=ti.f64),
        temperature: float,
    ):

        surface_tension = ti.types.ndarray(dtype=ti.f64)
        density = ti.types.ndarray(dtype=ti.f64)
        result = ti.types.ndarray(dtype=ti.f64)
        surface_tension[0] = self.effective_surface_tension(mass_concentration)
        density[0] = self.effective_density(mass_concentration)

        kget_kelvin_radius(
            surface_tension, density, molar_mass, float(temperature), result
        )
        return result

    def kelvin_term(
            self,
            radius: ti.types.ndarray(dtype=ti.f64),
            molar_mass: ti.types.ndarray(dtype=ti.f64),
            mass_concentration: ti.types.ndarray(dtype=ti.f64),
            temperature: float,
        ):
        r_k = self.kelvin_radius(molar_mass, mass_concentration, temperature)
        results = ti.types.ndarray(dtype=ti.f64)
        kget_kelvin_term(radius, r_k, results)
        return results
