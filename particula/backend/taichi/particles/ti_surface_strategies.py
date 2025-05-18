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
from particula.backend.taichi.particles.properties.ti_kelvin_effect_module import (
    kget_kelvin_radius,
    kget_kelvin_term,
)

def store(arr):
    # returns (scalar/1-D field, length)
    if isinstance(arr, (float, int)):
        f = ti.field(dtype=ti.f64, shape=())
        f[None] = float(arr)
        return f, 1
    # python list / tuple
    if not hasattr(arr, "shape"):
        n = len(arr)
        f = ti.field(dtype=ti.f64, shape=n)
        for i in range(n):
            f[i] = float(arr[i])
        return f, n
    # ti.types.ndarray or numpy.ndarray (no NumPy ops used)
    n = arr.shape[0]
    f = ti.field(dtype=ti.f64, shape=n)

    @ti.kernel
    def _copy(src: ti.types.ndarray(dtype=ti.f64, ndim=1)):
        for i in range(n):
            f[i] = src[i]

    _copy(arr)
    return f, n

def ensure_ti_nd(x):
    if isinstance(x, (float, int)):
        a = ti_nd(dtype=ti.f64, shape=(1,))
        a[0] = float(x)
        return a
    if not hasattr(x, "shape"):
        n = len(x)
        a = ti_nd(dtype=ti.f64, shape=(n,))
        for i in range(n):
            a[i] = float(x[i])
        return a
    return x   # already ndarray-like

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

    def __init__(self, surface_tension, density):
        self.surface_tension, self.n_species = store(surface_tension)
        self.density, _ = store(density)
        # ones array used for “mass” strategy
        self.ones = ti.field(dtype=ti.f64, shape=self.n_species)
        for i in range(self.n_species):
            self.ones[i] = 1.0

    @ti.kernel
    def weighted_average(
        self,
        values: ti.template(),
        weights: ti_nd(dtype=ti.f64, ndim=1),
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
class SurfaceStrategyMolar(SurfaceMixin):
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
        surface_tension=0.072,
        density=1000,
        molar_mass=0.01815,
    ):
        super().__init__(surface_tension, density)
        self.molar_mass, _ = store(molar_mass)

    def effective_surface_tension(self, mass_concentration):
        mc = ensure_ti_nd(mass_concentration)
        return float(self.weighted_average(self.surface_tension, mc, self.molar_mass))

    def effective_density(self, mass_concentration):
        mc = ensure_ti_nd(mass_concentration)
        return float(self.weighted_average(self.density, mc, self.molar_mass))

    def kelvin_radius(self, molar_mass, mass_concentration, temperature):
        M = ensure_ti_nd(molar_mass)
        mc = ensure_ti_nd(mass_concentration)
        st = ti_nd(dtype=ti.f64, shape=(1,))
        rho = ti_nd(dtype=ti.f64, shape=(1,))
        st[0] = self.effective_surface_tension(mc)
        rho[0] = self.effective_density(mc)
        return kget_kelvin_radius(st, rho, M, float(temperature))

    def kelvin_term(self, radius, molar_mass, mass_concentration, temperature):
        r = ensure_ti_nd(radius)
        M = ensure_ti_nd(molar_mass)
        mc = ensure_ti_nd(mass_concentration)
        r_k = self.kelvin_radius(M, mc, temperature)
        return kget_kelvin_term(r, r_k)


@ti.data_oriented
class SurfaceStrategyMass(SurfaceMixin):
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
        surface_tension=0.072,
        density=1000,
    ):
        super().__init__(surface_tension, density)

    def effective_surface_tension(self, mass_concentration):
        mc = ensure_ti_nd(mass_concentration)
        return float(self.weighted_average(self.surface_tension, mc, self.ones))

    def effective_density(self, mass_concentration):
        mc = ensure_ti_nd(mass_concentration)
        return float(self.weighted_average(self.density, mc, self.ones))

    def kelvin_radius(self, molar_mass, mass_concentration, temperature):
        M = ensure_ti_nd(molar_mass)
        mc = ensure_ti_nd(mass_concentration)
        st = ti_nd(dtype=ti.f64, shape=(1,))
        rho = ti_nd(dtype=ti.f64, shape=(1,))
        st[0] = self.effective_surface_tension(mc)
        rho[0] = self.effective_density(mc)
        return kget_kelvin_radius(st, rho, M, float(temperature))

    def kelvin_term(self, radius, molar_mass, mass_concentration, temperature):
        r = ensure_ti_nd(radius)
        M = ensure_ti_nd(molar_mass)
        mc = ensure_ti_nd(mass_concentration)
        r_k = self.kelvin_radius(M, mc, temperature)
        return kget_kelvin_term(r, r_k)


@ti.data_oriented
class SurfaceStrategyVolume(SurfaceMixin):
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
        surface_tension=0.072,
        density=1000,
    ):
        super().__init__(surface_tension, density)

    def effective_surface_tension(self, mass_concentration):
        mc = ensure_ti_nd(mass_concentration)
        return float(self.weighted_average(self.surface_tension, mc, self.density))

    def effective_density(self, mass_concentration):
        mc = ensure_ti_nd(mass_concentration)
        return float(self.weighted_average(self.density, mc, self.density))

    def kelvin_radius(self, molar_mass, mass_concentration, temperature):
        M = ensure_ti_nd(molar_mass)
        mc = ensure_ti_nd(mass_concentration)
        st = ti_nd(dtype=ti.f64, shape=(1,))
        rho = ti_nd(dtype=ti.f64, shape=(1,))
        st[0] = self.effective_surface_tension(mc)
        rho[0] = self.effective_density(mc)
        return kget_kelvin_radius(st, rho, M, float(temperature))

    def kelvin_term(self, radius, molar_mass, mass_concentration, temperature):
        r = ensure_ti_nd(radius)
        M = ensure_ti_nd(molar_mass)
        mc = ensure_ti_nd(mass_concentration)
        r_k = self.kelvin_radius(M, mc, temperature)
        return kget_kelvin_term(r, r_k)
