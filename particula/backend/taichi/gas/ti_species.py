"""Taichi implementation of particula.gas.species.GasSpecies.

Provides a Taichi-accelerated class for gas species, supporting
vectorized vapor pressure and concentration operations.

Examples:
    from particula.backend.taichi.gas import ti_create_gas_species
    gas = ti_create_gas_species("H2O", 0.018, concentration=1.0)
"""

import taichi as ti
import taichi.math as tim
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
ti.init(default_fp=ti.f64)  # safe default


@ti.data_oriented
class TiGasSpecies:

    @ti.kernel
    def _copy_field_kernel(self, src: ti.template(), dst: ti.types.ndarray()):
        for i in src:
            dst[i] = src[i]

    @ti.kernel
    def _add_concentration_kernel(self, delta: ti.types.ndarray()):
        for i in self.concentration:
            new_val = self.concentration[i] + delta[i]
            self.concentration[i] = ti.select(new_val < 0.0, 0.0, new_val)

    @ti.kernel
    def _set_concentration_kernel(self, new_vals: ti.types.ndarray()):
        for i in self.concentration:
            self.concentration[i] = ti.select(new_vals[i] < 0.0, 0.0, new_vals[i])
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
        if concentration_array.size == 1 and molar_mass_array.size > 1:
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
        self,
        temperature: ti.f64,
        result: ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        """
        Compute the pure-component vapor pressure for each species.

        Arguments:
            - temperature : Temperature in K.
            - result      : 1-D float64 NumPy array that will be filled
                            in-place with the vapor-pressure values [Pa].

        Returns:
            - None (results are written to ``result``).
        """
        for species_index, strategy in ti.static(
            enumerate(self.vapor_pressure_strategies)
        ):
            result[species_index] = strategy.fget_pure_vapor_pressure(
                temperature
            )

    @ti.kernel
    def _partial_pressure_kernel(
        self,
        temperature: ti.f64,
        result: ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        """
        Compute the partial pressure for each species.

        Arguments:
            - temperature : Temperature in K.
            - result      : 1-D float64 NumPy array that will be filled
                            in-place with the partial-pressure values [Pa].

        Returns:
            - None (results are written to ``result``).
        """
        for species_index, strategy in ti.static(
            enumerate(self.vapor_pressure_strategies)
        ):
            result[species_index] = strategy.fget_partial_pressure_internal(
                self.concentration[species_index],
                self.molar_mass[species_index],
                temperature,
            )

    @ti.kernel
    def _saturation_ratio_kernel(
        self,
        temperature: ti.f64,
        result: ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        """
        Compute the saturation ratio for each species.

        Arguments:
            - temperature : Temperature in K.
            - result      : 1-D float64 NumPy array that will be filled
                            in-place with the saturation ratio values.

        Returns:
            - None (results are written to ``result``).
        """
        for species_index, strategy in ti.static(
            enumerate(self.vapor_pressure_strategies)
        ):
            vapor_pressure = strategy.fget_pure_vapor_pressure(temperature)
            partial_pressure = strategy.fget_partial_pressure_internal(
                self.concentration[species_index],
                self.molar_mass[species_index],
                temperature,
            )
            result[species_index] = partial_pressure / vapor_pressure

    @ti.kernel
    def _saturation_concentration_kernel(
        self,
        temperature: ti.f64,
        result: ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        """
        Compute the saturation concentration for each species.

        Arguments:
            - temperature : Temperature in K.
            - result      : 1-D float64 NumPy array that will be filled
                            in-place with the saturation concentration [kg m⁻³].

        Returns:
            - None (results are written to ``result``).
        """
        for species_index, strategy in ti.static(
            enumerate(self.vapor_pressure_strategies)
        ):
            vapor_pressure = strategy.fget_pure_vapor_pressure(temperature)
            result[species_index] = (
                strategy.fget_concentration_from_pressure_internal(
                    vapor_pressure,
                    self.molar_mass[species_index],
                    temperature,
                )
            )

    def get_molar_mass(self):
        buf = ti.ndarray(dtype=ti.f64, shape=(self.n_species,))
        self._copy_field_kernel(self.molar_mass, buf)
        return buf

    def get_concentration(self):
        buf = ti.ndarray(dtype=ti.f64, shape=(self.n_species,))
        self._copy_field_kernel(self.concentration, buf)
        return buf

    def get_pure_vapor_pressure(self, temperature):
        buf = ti.ndarray(dtype=ti.f64, shape=(self.n_species,))
        self._pure_vapor_pressure_kernel(float(temperature), buf)
        return buf

    def get_partial_pressure(self, temperature):
        buf = ti.ndarray(dtype=ti.f64, shape=(self.n_species,))
        self._partial_pressure_kernel(float(temperature), buf)
        return buf

    def get_saturation_ratio(self, temperature):
        buf = ti.ndarray(dtype=ti.f64, shape=(self.n_species,))
        self._saturation_ratio_kernel(float(temperature), buf)
        return buf

    def get_saturation_concentration(self, temperature):
        buf = ti.ndarray(dtype=ti.f64, shape=(self.n_species,))
        self._saturation_concentration_kernel(float(temperature), buf)
        return buf

    def add_concentration(self, delta):
        arr_np = np.asarray(delta, dtype=np.float64)
        if arr_np.size == 1:
            arr_np = np.full(self.n_species, arr_np.item(), dtype=np.float64)
        if arr_np.size != self.n_species:
            raise ValueError("delta length mismatch")
        arr_ti = ti.ndarray(dtype=ti.f64, shape=(self.n_species,))
        arr_ti.from_numpy(arr_np)
        self._add_concentration_kernel(arr_ti)

    def set_concentration(self, new_value):
        arr_np = np.asarray(new_value, dtype=np.float64)
        if arr_np.size == 1:
            arr_np = np.full(self.n_species, arr_np.item(), dtype=np.float64)
        if arr_np.size != self.n_species:
            raise ValueError("new_value length mismatch")
        arr_ti = ti.ndarray(dtype=ti.f64, shape=(self.n_species,))
        arr_ti.from_numpy(arr_np)
        self._set_concentration_kernel(arr_ti)

    # meta dunders (str / len / + / +=) can remain python-side only


@register("GasSpecies", backend="taichi")
def ti_create_gas_species(*args, **kwargs):  # noqa: D401
    """
    Factory helper that instantiates :class:`GasSpecies` through the
    backend-dispatch mechanism.

    Arguments:
        - *args, **kwargs : Forwarded verbatim to ``GasSpecies``.
    Returns:
        - GasSpecies : A newly constructed Taichi ``GasSpecies`` instance.

    Examples:
        ```py
        from particula.backend.dispatch_register import use_backend
        use_backend(name="taichi")          # ensure Taichi backend
        gas = ti_create_gas_species("H2O", 0.018, concentration=1.0)
        ```
    """
    return TiGasSpecies(*args, **kwargs)


__all__ = ["TiGasSpecies", "ti_create_gas_species"]
