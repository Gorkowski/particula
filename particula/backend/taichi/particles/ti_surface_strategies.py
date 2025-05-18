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

High-level objectives:
    - Provide fast, vectorized, and Taichi-accelerated computation
      of surface property mixing and Kelvin effect terms.
    - Maintain API compatibility with the Python backend.
    - Support both scalar and array-valued properties.

References:
    - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric Chemistry
      and Physics: From Air Pollution to Climate Change (3rd ed.).
      Wiley.
    - "Kelvin equation," Wikipedia.
      https://en.wikipedia.org/wiki/Kelvin_equation

"""

import taichi as ti
from taichi.types import ndarray as ti_nd
import numpy as np
from typing import Union
from numpy.typing import NDArray

from particula.backend.taichi.particles.properties import (
        ti_get_kelvin_radius,  # <- NEW
        ti_get_kelvin_term,    # <- NEW
        fget_kelvin_radius,    # still used in in-kernel helpers
        fget_kelvin_term,
        kget_kelvin_radius,
        kget_kelvin_term,
        fget_ideal_activity_mass,
        kget_ideal_activity_mass,
        fget_ideal_activity_molar,
        kget_ideal_activity_molar,
        fget_ideal_activity_volume,
        kget_ideal_activity_volume,
        fget_water_activity_from_kappa_row,
        kget_water_volume_in_mixture,
)


@ti.data_oriented
class _SurfaceMixin:
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

    def __init__(self, surface_tension, density):
        self.surface_tension, self.n_species = _store(surface_tension)
        self.density, _ = _store(density)

    @ti.kernel
    def _weighted_average(
        self,
        values: ti.template(),
        weights: ti_nd(dtype=ti.f64, ndim=1),
        normalize_by: ti_nd(dtype=ti.f64, ndim=1) = None
    ) -> ti.f64:
        tot = 0.0
        for i in range(self.n_species):
            w = weights[i] if normalize_by is None else weights[i] / normalize_by[i]
            tot += w
        acc = 0.0
        for i in range(self.n_species):
            w = weights[i] if normalize_by is None else weights[i] / normalize_by[i]
            acc += values[i] * w / tot
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
class SurfaceStrategyMolar(_SurfaceMixin):
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

    Examples:
        >>> strat = SurfaceStrategyMolar([0.072, 0.073], [1000, 1200], [0.018, 0.02])
        >>> strat.effective_surface_tension([0.5, 0.5])
        0.0725
        >>> strat.kelvin_radius([0.018, 0.02], [0.5, 0.5], 298.15)
        array([...])

    References:
        - Seinfeld & Pandis (2016), Atmospheric Chemistry and Physics.
        - "Kelvin equation," Wikipedia.
    """

    def __init__(
        self,
        surface_tension: Union[float, NDArray[np.float64]] = 0.072,
        density: Union[float, NDArray[np.float64]] = 1000,
        molar_mass: Union[float, NDArray[np.float64]] = 0.01815,
    ):
        super().__init__(surface_tension, density)
        self.molar_mass, _ = _store(molar_mass)

    def effective_surface_tension(
        self, mass_concentration: ti_nd(dtype=ti.f64, ndim=1)
    ) -> float:
        return float(self._weighted_average(self.surface_tension, mass_concentration, self.molar_mass))

    def effective_density(
        self, mass_concentration: ti_nd(dtype=ti.f64, ndim=1)
    ) -> float:
        return float(self._weighted_average(self.density, mass_concentration, self.molar_mass))

    def kelvin_radius(
        self,
        molar_mass: ti_nd(dtype=ti.f64, ndim=1),
        mass_concentration: ti_nd(dtype=ti.f64, ndim=1),
        temperature: float,
    ):
        st = ti_nd(dtype=ti.f64, shape=(1,))
        rho = ti_nd(dtype=ti.f64, shape=(1,))
        st[0] = self.effective_surface_tension(mass_concentration)
        rho[0] = self.effective_density(mass_concentration)
        r_k = kget_kelvin_radius(st, rho, molar_mass, float(temperature))
        return r_k

    def kelvin_term(
        self,
        radius: ti_nd(dtype=ti.f64, ndim=1),
        molar_mass: ti_nd(dtype=ti.f64, ndim=1),
        mass_concentration: ti_nd(dtype=ti.f64, ndim=1),
        temperature: float,
    ):
        r_k = self.kelvin_radius(molar_mass, mass_concentration, temperature)
        return kget_kelvin_term(radius, r_k)


@ti.data_oriented
class SurfaceStrategyMass(_SurfaceMixin):
    """
    Surface property mixing by mass fraction.

    Computes effective surface tension and density using mass-fraction
    weighting. Provides Taichi-accelerated Kelvin radius and Kelvin term
    calculations.

    Attributes:
        - surface_tension : Taichi field for surface tension.
        - density : Taichi field for density.
        - n_species : Number of species/components.

    Methods:
        - effective_surface_tension: Mass-fraction weighted surface tension.
        - effective_density: Mass-fraction weighted density.
        - kelvin_radius: Kelvin radius (rₖ = (2σM)/(ρRT)).
        - kelvin_term: Kelvin term (exp(rₖ/r)).

    Examples:
        >>> strat = SurfaceStrategyMass([0.072, 0.073], [1000, 1200])
        >>> strat.effective_surface_tension([0.5, 0.5])
        0.0725

    References:
        - Seinfeld & Pandis (2016), Atmospheric Chemistry and Physics.
        - "Kelvin equation," Wikipedia.
    """

    def __init__(
        self,
        surface_tension: Union[float, NDArray[np.float64]] = 0.072,
        density: Union[float, NDArray[np.float64]] = 1000,
    ):
        super().__init__(surface_tension, density)

    def effective_surface_tension(
        self, mass_concentration: ti_nd(dtype=ti.f64, ndim=1)
    ) -> float:
        return float(self._weighted_average(self.surface_tension, mass_concentration, None))

    def effective_density(
        self, mass_concentration: ti_nd(dtype=ti.f64, ndim=1)
    ) -> float:
        return float(self._weighted_average(self.density, mass_concentration, None))

    def kelvin_radius(
        self,
        molar_mass: ti_nd(dtype=ti.f64, ndim=1),
        mass_concentration: ti_nd(dtype=ti.f64, ndim=1),
        temperature: float,
    ):
        st = ti_nd(dtype=ti.f64, shape=(1,))
        rho = ti_nd(dtype=ti.f64, shape=(1,))
        st[0] = self.effective_surface_tension(mass_concentration)
        rho[0] = self.effective_density(mass_concentration)
        r_k = kget_kelvin_radius(st, rho, molar_mass, float(temperature))
        return r_k

    def kelvin_term(
        self,
        radius: ti_nd(dtype=ti.f64, ndim=1),
        molar_mass: ti_nd(dtype=ti.f64, ndim=1),
        mass_concentration: ti_nd(dtype=ti.f64, ndim=1),
        temperature: float,
    ):
        r_k = self.kelvin_radius(molar_mass, mass_concentration, temperature)
        return kget_kelvin_term(radius, r_k)


@ti.data_oriented
class SurfaceStrategyVolume(_SurfaceMixin):
    """
    Surface property mixing by volume fraction.

    Computes effective surface tension and density using volume-fraction
    weighting. Provides Taichi-accelerated Kelvin radius and Kelvin term
    calculations.

    Attributes:
        - surface_tension : Taichi field for surface tension.
        - density : Taichi field for density.
        - n_species : Number of species/components.

    Methods:
        - effective_surface_tension: Volume-fraction weighted surface tension.
        - effective_density: Volume-fraction weighted density.
        - kelvin_radius: Kelvin radius (rₖ = (2σM)/(ρRT)).
        - kelvin_term: Kelvin term (exp(rₖ/r)).

    Examples:
        >>> strat = SurfaceStrategyVolume([0.072, 0.073], [1000, 1200])
        >>> strat.effective_surface_tension([0.5, 0.5])
        0.0725

    References:
        - Seinfeld & Pandis (2016), Atmospheric Chemistry and Physics.
        - "Kelvin equation," Wikipedia.
    """

    def __init__(
        self,
        surface_tension: Union[float, NDArray[np.float64]] = 0.072,
        density: Union[float, NDArray[np.float64]] = 1000,
    ):
        super().__init__(surface_tension, density)

    def effective_surface_tension(
        self, mass_concentration: ti_nd(dtype=ti.f64, ndim=1)
    ) -> float:
        return float(self._weighted_average(self.surface_tension, mass_concentration, self.density))

    def effective_density(
        self, mass_concentration: ti_nd(dtype=ti.f64, ndim=1)
    ) -> float:
        return float(self._weighted_average(self.density, mass_concentration, self.density))

    def kelvin_radius(
        self,
        molar_mass: ti_nd(dtype=ti.f64, ndim=1),
        mass_concentration: ti_nd(dtype=ti.f64, ndim=1),
        temperature: float,
    ):
        st = ti_nd(dtype=ti.f64, shape=(1,))
        rho = ti_nd(dtype=ti.f64, shape=(1,))
        st[0] = self.effective_surface_tension(mass_concentration)
        rho[0] = self.effective_density(mass_concentration)
        r_k = kget_kelvin_radius(st, rho, molar_mass, float(temperature))
        return r_k

    def kelvin_term(
        self,
        radius: ti_nd(dtype=ti.f64, ndim=1),
        molar_mass: ti_nd(dtype=ti.f64, ndim=1),
        mass_concentration: ti_nd(dtype=ti.f64, ndim=1),
        temperature: float,
    ):
        r_k = self.kelvin_radius(molar_mass, mass_concentration, temperature)
        return kget_kelvin_term(radius, r_k)
