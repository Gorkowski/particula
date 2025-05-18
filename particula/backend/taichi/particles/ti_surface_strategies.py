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

    @ti.func
    def _kelvin_radius_elem(
        self,
        surface_tension: ti.f64,
        density: ti.f64,
        molar_mass: ti.f64,
        temperature: ti.f64,
    ) -> ti.f64:
        # Delegates to the validated helper – keeps one single source of truth
        return fget_kelvin_radius(
            surface_tension, density, molar_mass, temperature
        )

    @ti.func
    def _kelvin_term_elem(
        self,
        particle_radius: ti.f64,
        kelvin_radius_value: ti.f64,
    ) -> ti.f64:
        return fget_kelvin_term(particle_radius, kelvin_radius_value)

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
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> float:
        """
        Compute effective surface tension by mole-fraction weighting.

        Arguments:
            - mass_concentration : Mass concentration(s) of each component.

        Returns:
            - effective_surface_tension : Weighted surface tension (float).

        Examples:
            >>> strat = SurfaceStrategyMolar([0.072, 0.073], [1000, 1200], [0.018, 0.02])
            >>> strat.effective_surface_tension([0.5, 0.5])
            0.0725
        """
        if self.n_species == 1:
            return self.surface_tension[None]
        mass_concentration_array = np.asarray(
            mass_concentration, dtype=np.float64
        )
        molar_mass_array = self.molar_mass.to_numpy()
        mole_fraction = mass_concentration_array / molar_mass_array
        mole_fraction /= mole_fraction.sum()
        return (
            self.surface_tension.to_numpy() * mole_fraction
        ).sum(dtype=np.float64)

    def effective_density(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> float:
        """
        Compute effective density by mole-fraction weighting.

        Arguments:
            - mass_concentration : Mass concentration(s) of each component.

        Returns:
            - effective_density : Weighted density (float).

        Examples:
            >>> strat = SurfaceStrategyMolar([0.072, 0.073], [1000, 1200], [0.018, 0.02])
            >>> strat.effective_density([0.5, 0.5])
            1100.0
        """
        if self.n_species == 1:
            return self.density[None]
        mass_concentration_array = np.asarray(
            mass_concentration, dtype=np.float64
        )
        molar_mass_array = self.molar_mass.to_numpy()
        mole_fraction = mass_concentration_array / molar_mass_array
        mole_fraction /= mole_fraction.sum()
        return (
            self.density.to_numpy() * mole_fraction
        ).sum(dtype=np.float64)

    def kelvin_radius(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        mass_concentration: Union[float, NDArray[np.float64]],
        temperature: float,
    ) -> Union[float, NDArray[np.float64]]:
        """
        Compute the Kelvin radius for the mixture.

        The Kelvin radius is given by:
            rₖ = (2 σ M) / (ρ R T)
        where:
            - σ : Effective surface tension
            - M : Molar mass
            - ρ : Effective density
            - R : Gas constant
            - T : Temperature

        Arguments:
            - molar_mass : Molar mass(es) of each component.
            - mass_concentration : Mass concentration(s) of each component.
            - temperature : Temperature in K.

        Returns:
            - kelvin_radius : Kelvin radius (float or ndarray).

        Examples:
            >>> strat = SurfaceStrategyMolar()
            >>> strat.kelvin_radius(0.018, 0.5, 298.15)
            array([...])

        References:
            - Seinfeld & Pandis (2016), Eq. 8.7.
        """
        surface_tension_value = self.effective_surface_tension(mass_concentration)
        effective_density_value = self.effective_density(mass_concentration)
        surface_tension_array = np.atleast_1d(
            surface_tension_value
        ).astype(np.float64)
        density_array = np.atleast_1d(
            effective_density_value
        ).astype(np.float64)
        molar_mass_array = np.atleast_1d(
            molar_mass
        ).astype(np.float64)
        return ti_get_kelvin_radius(
            surface_tension_array, density_array, molar_mass_array, float(temperature)
        )

    def kelvin_term(
        self,
        radius: Union[float, NDArray[np.float64]],
        molar_mass: Union[float, NDArray[np.float64]],
        mass_concentration: Union[float, NDArray[np.float64]],
        temperature: float,
    ) -> Union[float, NDArray[np.float64]]:
        """
        Compute the Kelvin term for the mixture.

        The Kelvin term is given by:
            exp(rₖ / r)
        where rₖ is the Kelvin radius and r is the particle radius.

        Arguments:
            - radius : Particle radius (float or ndarray).
            - molar_mass : Molar mass(es) of each component.
            - mass_concentration : Mass concentration(s) of each component.
            - temperature : Temperature in K.

        Returns:
            - kelvin_term : Kelvin term (float or ndarray).

        Examples:
            >>> strat = SurfaceStrategyMolar()
            >>> strat.kelvin_term(1e-7, 0.018, 0.5, 298.15)
            array([...])
        """
        kelvin_radius_value = self.kelvin_radius(
            molar_mass, mass_concentration, temperature
        )
        particle_radius_array = np.atleast_1d(
            radius
        ).astype(np.float64)
        kelvin_radius_array = np.atleast_1d(
            kelvin_radius_value
        ).astype(np.float64)
        return ti_get_kelvin_term(particle_radius_array, kelvin_radius_array)


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
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> float:
        """
        Compute effective surface tension by mass-fraction weighting.

        Arguments:
            - mass_concentration : Mass concentration(s) of each component.

        Returns:
            - effective_surface_tension : Weighted surface tension (float).

        Examples:
            >>> strat = SurfaceStrategyMass([0.072, 0.073], [1000, 1200])
            >>> strat.effective_surface_tension([0.5, 0.5])
            0.0725
        """
        if self.n_species == 1:
            return self.surface_tension[None]
        mass_concentration_array = np.asarray(
            mass_concentration, dtype=np.float64
        )
        mass_fraction = (
            mass_concentration_array / mass_concentration_array.sum()
        )
        return (
            self.surface_tension.to_numpy() * mass_fraction
        ).sum(dtype=np.float64)

    def effective_density(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> float:
        """
        Compute effective density by mass-fraction weighting.

        Arguments:
            - mass_concentration : Mass concentration(s) of each component.

        Returns:
            - effective_density : Weighted density (float).

        Examples:
            >>> strat = SurfaceStrategyMass([0.072, 0.073], [1000, 1200])
            >>> strat.effective_density([0.5, 0.5])
            1100.0
        """
        if self.n_species == 1:
            return self.density[None]
        mass_concentration_array = np.asarray(
            mass_concentration, dtype=np.float64
        )
        mass_fraction = (
            mass_concentration_array / mass_concentration_array.sum()
        )
        return (
            self.density.to_numpy() * mass_fraction
        ).sum(dtype=np.float64)

    def kelvin_radius(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        mass_concentration: Union[float, NDArray[np.float64]],
        temperature: float,
    ) -> Union[float, NDArray[np.float64]]:
        """
        Compute the Kelvin radius for the mixture.

        The Kelvin radius is given by:
            rₖ = (2 σ M) / (ρ R T)
        where:
            - σ : Effective surface tension
            - M : Molar mass
            - ρ : Effective density
            - R : Gas constant
            - T : Temperature

        Arguments:
            - molar_mass : Molar mass(es) of each component.
            - mass_concentration : Mass concentration(s) of each component.
            - temperature : Temperature in K.

        Returns:
            - kelvin_radius : Kelvin radius (float or ndarray).

        Examples:
            >>> strat = SurfaceStrategyMass()
            >>> strat.kelvin_radius(0.018, 0.5, 298.15)
            array([...])

        References:
            - Seinfeld & Pandis (2016), Eq. 8.7.
        """
        surface_tension_value = self.effective_surface_tension(mass_concentration)
        effective_density_value = self.effective_density(mass_concentration)
        surface_tension_array = np.atleast_1d(
            surface_tension_value
        ).astype(np.float64)
        density_array = np.atleast_1d(
            effective_density_value
        ).astype(np.float64)
        molar_mass_array = np.atleast_1d(
            molar_mass
        ).astype(np.float64)
        return ti_get_kelvin_radius(
            surface_tension_array, density_array, molar_mass_array, float(temperature)
        )

    def kelvin_term(
        self,
        radius: Union[float, NDArray[np.float64]],
        molar_mass: Union[float, NDArray[np.float64]],
        mass_concentration: Union[float, NDArray[np.float64]],
        temperature: float,
    ) -> Union[float, NDArray[np.float64]]:
        """
        Compute the Kelvin term for the mixture.

        The Kelvin term is given by:
            exp(rₖ / r)
        where rₖ is the Kelvin radius and r is the particle radius.

        Arguments:
            - radius : Particle radius (float or ndarray).
            - molar_mass : Molar mass(es) of each component.
            - mass_concentration : Mass concentration(s) of each component.
            - temperature : Temperature in K.

        Returns:
            - kelvin_term : Kelvin term (float or ndarray).

        Examples:
            >>> strat = SurfaceStrategyMass()
            >>> strat.kelvin_term(1e-7, 0.018, 0.5, 298.15)
            array([...])
        """
        kelvin_radius_value = self.kelvin_radius(
            molar_mass, mass_concentration, temperature
        )
        particle_radius_array = np.atleast_1d(
            radius
        ).astype(np.float64)
        kelvin_radius_array = np.atleast_1d(
            kelvin_radius_value
        ).astype(np.float64)
        return ti_get_kelvin_term(particle_radius_array, kelvin_radius_array)


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
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> float:
        """
        Compute effective surface tension by volume-fraction weighting.

        Arguments:
            - mass_concentration : Mass concentration(s) of each component.

        Returns:
            - effective_surface_tension : Weighted surface tension (float).

        Examples:
            >>> strat = SurfaceStrategyVolume([0.072, 0.073], [1000, 1200])
            >>> strat.effective_surface_tension([0.5, 0.5])
            0.0725
        """
        if self.n_species == 1:
            return self.surface_tension[None]
        mass_concentration_array = np.asarray(
            mass_concentration, dtype=np.float64
        )
        density_array = self.density.to_numpy()
        volume_array = mass_concentration_array / density_array
        volume_fraction_weight = volume_array / volume_array.sum()
        return (
            self.surface_tension.to_numpy() * volume_fraction_weight
        ).sum(dtype=np.float64)

    def effective_density(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> float:
        """
        Compute effective density by volume-fraction weighting.

        Arguments:
            - mass_concentration : Mass concentration(s) of each component.

        Returns:
            - effective_density : Weighted density (float).

        Examples:
            >>> strat = SurfaceStrategyVolume([0.072, 0.073], [1000, 1200])
            >>> strat.effective_density([0.5, 0.5])
            1100.0
        """
        if self.n_species == 1:
            return self.density[None]
        mass_concentration_array = np.asarray(
            mass_concentration, dtype=np.float64
        )
        density_array = self.density.to_numpy()
        volume_array = mass_concentration_array / density_array
        volume_fraction_weight = volume_array / volume_array.sum()
        return (
            self.density.to_numpy() * volume_fraction_weight
        ).sum(dtype=np.float64)

    def kelvin_radius(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        mass_concentration: Union[float, NDArray[np.float64]],
        temperature: float,
    ) -> Union[float, NDArray[np.float64]]:
        """
        Compute the Kelvin radius for the mixture.

        The Kelvin radius is given by:
            rₖ = (2 σ M) / (ρ R T)
        where:
            - σ : Effective surface tension
            - M : Molar mass
            - ρ : Effective density
            - R : Gas constant
            - T : Temperature

        Arguments:
            - molar_mass : Molar mass(es) of each component.
            - mass_concentration : Mass concentration(s) of each component.
            - temperature : Temperature in K.

        Returns:
            - kelvin_radius : Kelvin radius (float or ndarray).

        Examples:
            >>> strat = SurfaceStrategyVolume()
            >>> strat.kelvin_radius(0.018, 0.5, 298.15)
            array([...])

        References:
            - Seinfeld & Pandis (2016), Eq. 8.7.
        """
        surface_tension_value = self.effective_surface_tension(mass_concentration)
        effective_density_value = self.effective_density(mass_concentration)
        surface_tension_array = np.atleast_1d(
            surface_tension_value
        ).astype(np.float64)
        density_array = np.atleast_1d(
            effective_density_value
        ).astype(np.float64)
        molar_mass_array = np.atleast_1d(
            molar_mass
        ).astype(np.float64)
        return ti_get_kelvin_radius(
            surface_tension_array, density_array, molar_mass_array, float(temperature)
        )

    def kelvin_term(
        self,
        radius: Union[float, NDArray[np.float64]],
        molar_mass: Union[float, NDArray[np.float64]],
        mass_concentration: Union[float, NDArray[np.float64]],
        temperature: float,
    ) -> Union[float, NDArray[np.float64]]:
        """
        Compute the Kelvin term for the mixture.

        The Kelvin term is given by:
            exp(rₖ / r)
        where rₖ is the Kelvin radius and r is the particle radius.

        Arguments:
            - radius : Particle radius (float or ndarray).
            - molar_mass : Molar mass(es) of each component.
            - mass_concentration : Mass concentration(s) of each component.
            - temperature : Temperature in K.

        Returns:
            - kelvin_term : Kelvin term (float or ndarray).

        Examples:
            >>> strat = SurfaceStrategyVolume()
            >>> strat.kelvin_term(1e-7, 0.018, 0.5, 298.15)
            array([...])
        """
        kelvin_radius_value = self.kelvin_radius(
            molar_mass, mass_concentration, temperature
        )
        particle_radius_array = np.atleast_1d(
            radius
        ).astype(np.float64)
        kelvin_radius_array = np.atleast_1d(
            kelvin_radius_value
        ).astype(np.float64)
        return ti_get_kelvin_term(particle_radius_array, kelvin_radius_array)
