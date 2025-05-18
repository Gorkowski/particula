"""Taichi implementation of particula.particles.activity_strategies."""

import taichi as ti
import numpy as np
from typing import Union
from numpy.typing import NDArray
from particula.backend.dispatch_register import register

from particula.backend.taichi.particles.properties.ti_activity_module import (
    fget_surface_partial_pressure,
)

# ────────────────────────── shared mixin (no public ctor) ───────────────────
@ti.data_oriented
class _ActivityMixin:
    """Helpers common to every activity strategy."""

    # one-off element-wise helpers ------------------------------------------------
    @ti.func
    def fget_surface_partial_pressure(
        self,
        pure_vapor_pressure: ti.f64,
        activity: ti.f64,
    ) -> ti.f64:
        return fget_surface_partial_pressure(pure_vapor_pressure, activity)

    # ─────────────────────────── kernel helper (scalar) ────────────────────
    @ti.func
    def fget_partial_pressure_internal(          # NEW
        self,
        mass_concentration: ti.f64,
        pure_vapor_pressure: ti.f64,
    ) -> ti.f64:
        """
        Element-wise surface partial pressure used inside Taichi kernels.

        Returns p = a × p⁰.  Here we apply the simplest ideal fallback
        (activity = 1) so the value reduces to the pure vapor pressure.
        More sophisticated activity models can override this function in
        their concrete class if needed.
        """
        return pure_vapor_pressure   # ideal behaviour: activity ≡ 1

    # vectorised kernel (1-D ndarray in / out) -----------------------------------
    @ti.kernel
    def kget_surface_partial_pressure(
        self,
        pure_vapor_pressure: ti.types.ndarray(dtype=ti.f64, ndim=1),
        activity: ti.types.ndarray(dtype=ti.f64, ndim=1),
        result: ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        """
        Vectorised Raoult/Margules: p = a × p⁰ (element-wise).

        Arguments:
            - pure_vapor_pressure : 1D ndarray of pure vapor pressures.
            - activity : 1D ndarray of activity values.
            - result : 1D ndarray to store the computed partial pressures.

        Returns:
            - None. Results are written in-place to the result array.

        Equation:
            p = a × p⁰

        Examples:
            ```py
            result = np.empty_like(pure_vapor_pressure)
            self._kget_surface_partial_pressure(
                pure_vapor_pressure, activity, result
            )
            # result now contains the partial pressures
            ```
        """
        for i in ti.ndrange(result.shape[0]):
            result[i] = fget_surface_partial_pressure(
                pure_vapor_pressure[i], activity[i]
            )

    # public wrapper identical to NumPy API --------------------------------------
    def partial_pressure(
        self,
        pure_vapor_pressure: Union[float, NDArray[np.float64]],
        mass_concentration: Union[float, NDArray[np.float64]],
    ) -> Union[float, NDArray[np.float64]]:
        """
        Return surface vapor pressure given pure vapor pressure and concentration.

        Computes the surface vapor pressure using the formula:
            p = a × p⁰

        Arguments:
            - pure_vapor_pressure : float or 1D ndarray of pure vapor pressures.
            - mass_concentration : float or 1D ndarray of mass concentrations.

        Returns:
            - float or ndarray of surface vapor pressures.

        Examples:
            ```py
            strategy = ActivityIdealMolar(molar_mass=0.018)
            p = strategy.partial_pressure(1000.0, 0.5)
            # Output: 1000.0
            ```
        """
        if np.ndim(pure_vapor_pressure) == 0:  # scalar case
            # do the computation directly in Python to stay outside Taichi
            return float(pure_vapor_pressure) * float(
                self._activity_func(float(mass_concentration))
            )

        # vector case – reuse activity kernel then bulk multiply
        activity = self.activity(mass_concentration)  # ndarray (same shape)
        result = np.empty_like(pure_vapor_pressure, dtype=np.float64)
        self.kget_surface_partial_pressure(pure_vapor_pressure, activity, result)
        return result


# ───────────────────────────── concrete strategies ──────────────────────────
@ti.data_oriented
class ActivityIdealMolar(_ActivityMixin):
    """
    Taichi drop-in for ActivityIdealMolar.

    Implements the ideal molar activity strategy using Taichi kernels.
    Computes activity as the mole fraction of each component.

    Attributes:
        - molar_mass : Molar mass of the component.

    Methods:
        - activity : Compute activity (mole fraction) for given mass concentration.
        - partial_pressure : Compute surface vapor pressure.

    Examples:
        ```py
        strategy = ActivityIdealMolar(molar_mass=0.018)
        a = strategy.activity(np.array([0.5, 0.5]))
        # Output: array([0.5, 0.5])
        ```

    References:
        - "Mole fraction," [Wikipedia](https://en.wikipedia.org/wiki/Mole_fraction)
    """

    def __init__(self, molar_mass: float | NDArray[np.float64] = 0.0):
        """
        Initialize ActivityIdealMolar with molar mass.

        Arguments:
            - molar_mass : Molar mass of the component.

        Returns:
            - None
        """
        self.molar_mass = ti.field(ti.f64, shape=())
        self.molar_mass[None] = float(np.asarray(molar_mass))

    def _activity_func(self, mass_concentration: float) -> float:
        return 1.0

    @ti.kernel
    def kget_activity(
        self,
        mass_concentration: ti.types.ndarray(dtype=ti.f64, ndim=1),
        result: ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        """
        Compute activity (mole fraction) for each component.

        Arguments:
            - mass_concentration : 1D ndarray of mass concentrations.
            - result : 1D ndarray to store the computed activities.

        Returns:
            - None. Results are written in-place to the result array.

        Equation:
            a = (mass_concentration / molar_mass) / total_moles

        Examples:
            ```py
            result = np.empty_like(mass_concentration)
            self._kget_activity(mass_concentration, result)
            # result now contains the activities
            ```
        """
        molar_mass = self.molar_mass[None]
        total_moles = ti.f64(0.0)
        for i in range(mass_concentration.shape[0]):
            total_moles += mass_concentration[i] / molar_mass
        for i in range(mass_concentration.shape[0]):
            result[i] = (
                0.0
                if total_moles == 0.0
                else (mass_concentration[i] / molar_mass) / total_moles
            )

    def activity(self, mass_concentration):
        """
        Compute activity (mole fraction) for given mass concentration.

        Arguments:
            - mass_concentration : float or 1D ndarray of mass concentrations.

        Returns:
            - float or ndarray of activities.

        Examples:
            ```py
            strategy = ActivityIdealMolar(molar_mass=0.018)
            a = strategy.activity(np.array([0.5, 0.5]))
            # Output: array([0.5, 0.5])
            ```
        """
        if np.ndim(mass_concentration) == 0:
            return self._activity_func(float(mass_concentration))

        result = np.empty_like(mass_concentration, dtype=np.float64)
        self.kget_activity(mass_concentration, result)
        return result


@ti.data_oriented
class ActivityIdealMass(_ActivityMixin):
    """
    Taichi drop-in for ActivityIdealMass (parameter-free).

    Implements the ideal mass activity strategy using Taichi kernels.
    Computes activity as the mass fraction of each component.

    Methods:
        - activity : Compute activity (mass fraction) for given mass concentration.
        - partial_pressure : Compute surface vapor pressure.

    Examples:
        ```py
        strategy = ActivityIdealMass()
        a = strategy.activity(np.array([0.5, 0.5]))
        # Output: array([0.5, 0.5])
        ```

    References:
        - "Mass fraction (chemistry)," [Wikipedia](https://en.wikipedia.org/wiki/Mass_fraction_(chemistry))
    """

    def _activity_func(self, mass_concentration: float) -> float:
        return 1.0

    @ti.kernel
    def kget_activity(
        self,
        mass_concentration: ti.types.ndarray(dtype=ti.f64, ndim=1),
        result: ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        """
        Compute activity (mass fraction) for each component.

        Arguments:
            - mass_concentration : 1D ndarray of mass concentrations.
            - result : 1D ndarray to store the computed activities.

        Returns:
            - None. Results are written in-place to the result array.

        Equation:
            a = mass_concentration / total_mass

        Examples:
            ```py
            result = np.empty_like(mass_concentration)
            self._kget_activity(mass_concentration, result)
            # result now contains the activities
            ```
        """
        total_mass = ti.f64(0.0)
        for i in range(mass_concentration.shape[0]):
            total_mass += mass_concentration[i]
        for i in range(mass_concentration.shape[0]):
            result[i] = (
                0.0
                if total_mass == 0.0
                else mass_concentration[i] / total_mass
            )

    def activity(self, mass_concentration):
        """
        Compute activity (mass fraction) for given mass concentration.

        Arguments:
            - mass_concentration : float or 1D ndarray of mass concentrations.

        Returns:
            - float or ndarray of activities.

        Examples:
            ```py
            strategy = ActivityIdealMass()
            a = strategy.activity(np.array([0.5, 0.5]))
            # Output: array([0.5, 0.5])
            ```
        """
        if np.ndim(mass_concentration) == 0:
            return self._activity_func(float(mass_concentration))
        result = np.empty_like(mass_concentration, dtype=np.float64)
        self.kget_activity(mass_concentration, result)
        return result


@ti.data_oriented
class ActivityIdealVolume(_ActivityMixin):
    """
    Taichi drop-in for ActivityIdealVolume.

    Implements the ideal volume activity strategy using Taichi kernels.
    Computes activity as the volume fraction of each component.

    Attributes:
        - density : Density of the component.

    Methods:
        - activity : Compute activity (volume fraction) for given mass concentration.
        - partial_pressure : Compute surface vapor pressure.

    Examples:
        ```py
        strategy = ActivityIdealVolume(density=1000.0)
        a = strategy.activity(np.array([0.5, 0.5]))
        # Output: array([0.5, 0.5])
        ```

    References:
        - "Volume fraction," [Wikipedia](https://en.wikipedia.org/wiki/Volume_fraction)
    """

    def __init__(self, density: float | NDArray[np.float64] = 0.0):
        """
        Initialize ActivityIdealVolume with density.

        Arguments:
            - density : Density of the component.

        Returns:
            - None
        """
        self.density = ti.field(ti.f64, shape=())
        self.density[None] = float(np.asarray(density))

    def _activity_func(self, mass_concentration: float) -> float:
        return 1.0

    @ti.kernel
    def kget_activity(
        self,
        mass_concentration: ti.types.ndarray(dtype=ti.f64, ndim=1),
        result: ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        """
        Compute activity (volume fraction) for each component.

        Arguments:
            - mass_concentration : 1D ndarray of mass concentrations.
            - result : 1D ndarray to store the computed activities.

        Returns:
            - None. Results are written in-place to the result array.

        Equation:
            a = (mass_concentration / density) / total_volume

        Examples:
            ```py
            result = np.empty_like(mass_concentration)
            self._kget_activity(mass_concentration, result)
            # result now contains the activities
            ```
        """
        density = self.density[None]
        total_volume = ti.f64(0.0)
        for i in range(mass_concentration.shape[0]):
            total_volume += mass_concentration[i] / density
        for i in range(mass_concentration.shape[0]):
            result[i] = (
                0.0
                if total_volume == 0.0
                else (mass_concentration[i] / density) / total_volume
            )

    def activity(self, mass_concentration):
        """
        Compute activity (volume fraction) for given mass concentration.

        Arguments:
            - mass_concentration : float or 1D ndarray of mass concentrations.

        Returns:
            - float or ndarray of activities.

        Examples:
            ```py
            strategy = ActivityIdealVolume(density=1000.0)
            a = strategy.activity(np.array([0.5, 0.5]))
            # Output: array([0.5, 0.5])
            ```
        """
        if np.ndim(mass_concentration) == 0:
            return self._activity_func(float(mass_concentration))
        result = np.empty_like(mass_concentration, dtype=np.float64)
        self.kget_activity(mass_concentration, result)
        return result


@ti.data_oriented
class ActivityKappaParameter(_ActivityMixin):
    """
    Taichi drop-in for ActivityKappaParameter (non-ideal).

    Implements the non-ideal kappa-Köhler activity strategy using Taichi kernels.
    Computes water activity using the kappa parameterization.

    Attributes:
        - kappa : Kappa values for each species.
        - density : Densities for each species.
        - molar_mass : Molar masses for each species.
        - water_index : Index of the water species.

    Methods:
        - activity : Compute water activity for given mass concentration.
        - partial_pressure : Compute surface vapor pressure.

    Examples:
        ```py
        strategy = ActivityKappaParameter(
            kappa=np.array([0.5, 0.0]),
            density=np.array([1000.0, 1800.0]),
            molar_mass=np.array([0.018, 0.058]),
            water_index=0,
        )
        a = strategy.activity(np.array([0.5, 0.5]))
        # Output: array([...])
        ```

    References:
        - Petters, M. D., & Kreidenweis, S. M. (2007). A single parameter
          representation of hygroscopic growth and cloud condensation nucleus
          activity. Atmospheric Chemistry and Physics, 7(8), 1961–1971.
          [DOI](https://doi.org/10.5194/acp-7-1961-2007)
        - "Köhler theory," [Wikipedia](https://en.wikipedia.org/wiki/K%C3%B6hler_theory)
    """

    def __init__(
        self,
        kappa: NDArray[np.float64],
        density: NDArray[np.float64],
        molar_mass: NDArray[np.float64],
        water_index: int = 0,
    ):
        """
        Initialize ActivityKappaParameter with kappa, density, molar mass, and water index.

        Arguments:
            - kappa : 1D ndarray of kappa values for each species.
            - density : 1D ndarray of densities for each species.
            - molar_mass : 1D ndarray of molar masses for each species.
            - water_index : Index of the water species.

        Returns:
            - None
        """
        kappa = np.asarray(kappa, dtype=np.float64)
        density = np.asarray(density, dtype=np.float64)
        molar_mass = np.asarray(molar_mass, dtype=np.float64)

        n_species = kappa.size
        self.kappa = ti.field(ti.f64, shape=n_species)
        self.density = ti.field(ti.f64, shape=n_species)
        self.molar_mass = ti.field(ti.f64, shape=n_species)
        for i in range(n_species):
            self.kappa[i] = kappa[i]
            self.density[i] = density[i]
            self.molar_mass[i] = molar_mass[i]

        self.water_index = ti.field(dtype=ti.i32, shape=())
        self.water_index[None] = int(water_index)

    def _activity_func(self, mass_concentration: float) -> float:
        return 1.0

    @ti.kernel
    def kget_activity(
        self,
        mass_concentration: ti.types.ndarray(dtype=ti.f64, ndim=1),
        result: ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        """
        Compute water activity using the kappa-Köhler parameterization.

        Arguments:
            - mass_concentration : 1D ndarray of mass concentrations.
            - result : 1D ndarray to store the computed activities.

        Returns:
            - None. Results are written in-place to the result array.

        Equation:
            For water (index w):
            a_w = 1 / (1 + κ_mixed × (1 - φ_w) / φ_w)
            where φ_w = water volume fraction

        Examples:
            ```py
            result = np.empty_like(mass_concentration)
            self._kget_activity(mass_concentration, result)
            # result now contains the activities
            ```
        """
        n_species = mass_concentration.shape[0]
        water_index = self.water_index[None]

        # mole fractions -------------------------------------------------
        total_moles = ti.f64(0.0)
        for s in range(n_species):
            total_moles += mass_concentration[s] / self.molar_mass[s]
        for s in range(n_species):
            moles_species = mass_concentration[s] / self.molar_mass[s]
            result[s] = (
                0.0
                if total_moles == 0.0
                else moles_species / total_moles
            )

        # κ-Köhler water activity ---------------------------------------
        volume_sum = ti.f64(0.0)
        for s in range(n_species):
            volume_sum += mass_concentration[s] / self.density[s]

        water_volume_fraction = ti.f64(0.0)
        if volume_sum > 0.0:
            water_volume_fraction = (
                mass_concentration[water_index] / self.density[water_index]
            ) / volume_sum
        solute_volume_fraction = 1.0 - water_volume_fraction

        kappa_mixed = ti.f64(0.0)
        if solute_volume_fraction > 0.0:
            if n_species == 2:
                kappa_mixed = self.kappa[1 - water_index]
            else:
                for s in range(n_species):
                    if s != water_index:
                        volume_fraction_species = (
                            (mass_concentration[s] / self.density[s]) / volume_sum
                        )
                        kappa_mixed += (
                            (volume_fraction_species / solute_volume_fraction)
                            * self.kappa[s]
                        )

        volume_term = 0.0
        if water_volume_fraction > 0.0:
            volume_term = (
                kappa_mixed * solute_volume_fraction / water_volume_fraction
            )

        result[water_index] = (
            0.0
            if water_volume_fraction == 0.0
            else 1.0 / (1.0 + volume_term)
        )

    def activity(self, mass_concentration):
        """
        Compute water activity for given mass concentration.

        Arguments:
            - mass_concentration : float or 1D ndarray of mass concentrations.

        Returns:
            - float or ndarray of activities.

        Examples:
            ```py
            strategy = ActivityKappaParameter(
                kappa=np.array([0.5, 0.0]),
                density=np.array([1000.0, 1800.0]),
                molar_mass=np.array([0.018, 0.058]),
                water_index=0,
            )
            a = strategy.activity(np.array([0.5, 0.5]))
            # Output: array([...])
            ```
        """
        if np.ndim(mass_concentration) == 0:
            return self._activity_func(float(mass_concentration))
        result = np.empty_like(mass_concentration, dtype=np.float64)
        self.kget_activity(mass_concentration, result)
        return result
