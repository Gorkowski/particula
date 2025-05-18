"""Taichi implementation of particula.gas.species.GasSpecies.

Provides a Taichi-accelerated class for gas species, supporting
vectorized vapor pressure and concentration operations.

Examples:
    from particula.backend.taichi.gas import ti_create_gas_species
    gas = ti_create_gas_species("H2O", 0.018, concentration=1.0)
"""
import taichi as ti
import numpy as np
from typing import Union
from numpy.typing import NDArray

from particula.backend.dispatch_register import register
from particula.backend.taichi.gas.ti_vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
    AntoineVaporPressureStrategy,
    WaterBuckStrategy,
    ClausiusClapeyronStrategy,
)
# from particula.backend.taichi.gas.properties import (
#     fget_partial_pressure,          # element-wise helper
# )
ti.init(default_fp=ti.f64)          # safe default

@ti.data_oriented
class GasSpecies:
    """
    Taichi-based gas species container for vectorized property evaluation.

    Provides molar mass, concentration, and vapor pressure strategy
    management for one or more gas species using Taichi fields.

    Attributes:
        - name : Gas species name(s).
        - molar_mass : Ti field holding molar mass [kg mol⁻¹].
        - concentration : Ti field holding concentration [kg m⁻³].
        - vapor_pressure_strategies : List of VP strategies per species.
        - partitioning : Flag indicating whether species can partition.
        - n_species : Number of species.

    Methods:
        - get_molar_mass
        - get_concentration
        - get_pure_vapor_pressure
        - get_partial_pressure
        - get_saturation_ratio
        - get_saturation_concentration
        - add_concentration
        - set_concentration

    Examples:
        ```py
        from particula.backend.taichi.gas import ti_create_gas_species
        gas = ti_create_gas_species("H2O", 0.018, concentration=1.0)
        sat_ratio = gas.get_saturation_ratio(298.15)
        ```

    References:
        - "Antoine equation", Wikipedia.  (URL)
    """

    # ─── constructor ──────────────────────────────────────────────
    def __init__(
        self,
        name: Union[str, NDArray[np.str_]],
        molar_mass: Union[float, NDArray[np.float64]],
        vapor_pressure_strategy: Union[
            AntoineVaporPressureStrategy,
            WaterBuckStrategy,
            ClausiusClapeyronStrategy,
            ConstantVaporPressureStrategy,
        ] = ConstantVaporPressureStrategy(0.0),
        partitioning: bool = True,
        concentration: Union[float, NDArray[np.float64]] = 0.0,
    ):
        """
        Initialize GasSpecies.

        Arguments:
            - name : Species name(s).
            - molar_mass : Scalar or array in kg mol⁻¹.
            - vapor_pressure_strategy : Strategy instance or list
              (default ConstantVaporPressureStrategy(0)).
            - partitioning : Whether species can partition (default True).
            - concentration : Scalar/array concentration in kg m⁻³ (default 0).

        Returns:
            - None
        """
        self.name = name
        self.partitioning = partitioning

        molar_mass_array = np.atleast_1d(
            np.asarray(molar_mass, dtype=np.float64)
        )
        concentration_array = np.atleast_1d(
            np.asarray(concentration, dtype=np.float64)
        )
        if (concentration_array.size == 1
                and molar_mass_array.size > 1):
            concentration_array = np.full(
                molar_mass_array.size,
                concentration_array.item(),
                dtype=np.float64,
            )

        self.n_species = int(molar_mass_array.size)

        self.molar_mass = ti.field(ti.f64, shape=self.n_species)
        self.concentration = ti.field(ti.f64, shape=self.n_species)
        for species_index in range(self.n_species):
            self.molar_mass[species_index] = molar_mass_array[species_index]
            self.concentration[species_index] = max(
                0.0, concentration_array[species_index]
            )

        # make list of strategies the same length as species
        if not isinstance(vapor_pressure_strategy, list):
            self.vapor_pressure_strategies = [
                vapor_pressure_strategy
            ] * self.n_species
        elif len(vapor_pressure_strategy) == 1 and self.n_species > 1:
            self.vapor_pressure_strategies = (
                vapor_pressure_strategy * self.n_species
            )
        else:
            if len(vapor_pressure_strategy) != self.n_species:
                raise ValueError(
                    "len(vapor_pressure_strategy) must equal n_species"
                )
            self.vapor_pressure_strategies = vapor_pressure_strategy

    @ti.kernel
    def _pure_vapor_pressure_kernel(
        self, temperature: ti.f64,
        result: ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        for species_index, strategy in ti.static(
            enumerate(self.vapor_pressure_strategies)
        ):
            result[species_index] = strategy._pure_vp_func(temperature)

    @ti.kernel
    def _partial_pressure_kernel(
        self, temperature: ti.f64,
        result: ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        for species_index, strategy in ti.static(
            enumerate(self.vapor_pressure_strategies)
        ):
            result[species_index] = strategy._partial_pressure_func(
                self.concentration[species_index],
                self.molar_mass[species_index],
                temperature,
            )

    @ti.kernel
    def _saturation_ratio_kernel(
        self, temperature: ti.f64,
        result: ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        for species_index, strategy in ti.static(
            enumerate(self.vapor_pressure_strategies)
        ):
            vapor_pressure = strategy._pure_vp_func(temperature)
            partial_pressure = strategy._partial_pressure_func(
                self.concentration[species_index],
                self.molar_mass[species_index],
                temperature,
            )
            result[species_index] = (
                partial_pressure / vapor_pressure
            )

    @ti.kernel
    def _saturation_concentration_kernel(
        self, temperature: ti.f64,
        result: ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        for species_index, strategy in ti.static(
            enumerate(self.vapor_pressure_strategies)
        ):
            vapor_pressure = strategy._pure_vp_func(temperature)
            result[species_index] = strategy._concentration_func(
                vapor_pressure,
                self.molar_mass[species_index],
                temperature,
            )

    def get_molar_mass(self):
        """
        Return molar mass for each species.

        Returns:
            - Molar mass scalar or array [kg mol⁻¹].
        """
        return (self.molar_mass[0] if self.n_species == 1
                else self.molar_mass.to_numpy())

    def get_concentration(self):
        """
        Return concentration for each species.

        Returns:
            - Concentration scalar or array [kg m⁻³].
        """
        return (self.concentration[0] if self.n_species == 1
                else self.concentration.to_numpy())

    def get_pure_vapor_pressure(self, temperature):
        """
        Compute pure vapor pressure for each species.

        Arguments:
            - temperature : Temperature in K.

        Returns:
            - Pure vapor pressure scalar or array [Pa].
        """
        buffer = np.empty(self.n_species, dtype=np.float64)
        self._pure_vapor_pressure_kernel(float(temperature), buffer)
        return buffer[0] if self.n_species == 1 else buffer

    def get_partial_pressure(self, temperature):
        """
        Compute partial pressure for each species.

        Arguments:
            - temperature : Temperature in K.

        Returns:
            - Partial pressure scalar or array [Pa].
        """
        buffer = np.empty(self.n_species, dtype=np.float64)
        self._partial_pressure_kernel(float(temperature), buffer)
        return buffer[0] if self.n_species == 1 else buffer

    def get_saturation_ratio(self, temperature):
        """
        Compute saturation ratio for each species.

        Arguments:
            - temperature : Temperature in K.

        Returns:
            - Saturation ratio scalar or array.
        """
        buffer = np.empty(self.n_species, dtype=np.float64)
        self._saturation_ratio_kernel(float(temperature), buffer)
        return buffer[0] if self.n_species == 1 else buffer

    def get_saturation_concentration(self, temperature):
        """
        Compute saturation concentration for each species.

        Arguments:
            - temperature : Temperature in K.

        Returns:
            - Saturation concentration scalar or array [kg m⁻³].
        """
        buffer = np.empty(self.n_species, dtype=np.float64)
        self._saturation_concentration_kernel(float(temperature), buffer)
        return buffer[0] if self.n_species == 1 else buffer

    def add_concentration(self, delta):
        """
        Add delta to concentration for each species.

        Arguments:
            - delta : Scalar or array to add [kg m⁻³].

        Returns:
            - None
        """
        delta = np.atleast_1d(np.asarray(delta, dtype=np.float64))
        if delta.size == 1 and self.n_species > 1:
            delta = np.full(self.n_species, delta.item(), dtype=np.float64)
        if delta.size != self.n_species:
            raise ValueError("delta length mismatch")
        for species_index in range(self.n_species):
            self.concentration[species_index] = max(
                0.0,
                self.concentration[species_index] + delta[species_index],
            )

    def set_concentration(self, new_value):
        """
        Set concentration for each species.

        Arguments:
            - new_value : Scalar or array to set [kg m⁻³].

        Returns:
            - None
        """
        self.add_concentration(np.asarray(new_value, dtype=np.float64)
                               - self.get_concentration())

    # meta dunders (str / len / + / +=) can remain python-side only

@register("GasSpecies", backend="taichi")
def ti_create_gas_species(*args, **kwargs):     # noqa: D401
    """Factory so external code can request GasSpecies via backend."""
    return GasSpecies(*args, **kwargs)

__all__ = ["GasSpecies", "ti_create_gas_species"]
