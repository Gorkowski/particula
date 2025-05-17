"""Taichi implementation of particula.gas.species.GasSpecies."""
import taichi as ti
import numpy as np
from typing import Union
from numpy.typing import NDArray

from particula.backend.dispatch_register import register
from particula.backend.taichi.gas.ti_vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
    VaporPressureStrategy,          # re-exported by the Taichi file
)
from particula.backend.taichi.gas.properties.ti_pressure_function_module import (
    fget_partial_pressure,          # element-wise helper
    fget_saturation_ratio_from_pressure,
    fget_concentration_from_pressure,
)
ti.init(default_fp=ti.f64)          # safe default

@ti.data_oriented
class GasSpecies:
    """
    Taichi drop-in replacement for particula.gas.species.GasSpecies.
    Stores molar_mass & concentration in Ti fields and re-uses the
    existing Taichi vapor-pressure helpers.
    """

    # ─── constructor ──────────────────────────────────────────────
    def __init__(                         # keep original signature
        self,
        name: Union[str, NDArray[np.str_]],
        molar_mass: Union[float, NDArray[np.float64]],
        vapor_pressure_strategy: Union[VaporPressureStrategy,
                                       list[VaporPressureStrategy]]
            = ConstantVaporPressureStrategy(0.0),
        partitioning: bool = True,
        concentration: Union[float, NDArray[np.float64]] = 0.0,
    ):
        # python-side meta data
        self.name          = name
        self.partitioning  = partitioning

        # flatten → ndarray so we know the length
        mm_np   = np.atleast_1d(np.asarray(molar_mass,   dtype=np.float64))
        conc_np = np.atleast_1d(np.asarray(concentration, dtype=np.float64))
        if conc_np.size == 1 and mm_np.size > 1:
            conc_np = np.full(mm_np.size, conc_np.item(), dtype=np.float64)

        self.n_species = int(mm_np.size)

        # persistent Ti fields
        self.molar_mass   = ti.field(ti.f64, shape=self.n_species)
        self.concentration = ti.field(ti.f64, shape=self.n_species)
        for i in range(self.n_species):          # one-off copy
            self.molar_mass[i]    = mm_np[i]
            self.concentration[i] = max(0.0, conc_np[i])   # no negatives

        # make list of strategies the same length as species
        if not isinstance(vapor_pressure_strategy, list):
            self.strategies = [vapor_pressure_strategy] * self.n_species
        elif len(vapor_pressure_strategy) == 1 and self.n_species > 1:
            self.strategies = vapor_pressure_strategy * self.n_species
        else:
            if len(vapor_pressure_strategy) != self.n_species:
                raise ValueError("len(vapor_pressure_strategy) "
                                 "must equal #species")
            self.strategies = vapor_pressure_strategy

    # element-wise helper already exists → wrap in kernel
    @ti.kernel
    def _partial_pressure_kernel(
        self, temperature: ti.f64,
        result: ti.types.ndarray(dtype=ti.f64, ndim=1)
    ):
        """Vectorised partial pressure for every species."""
        for i in self.concentration:
            result[i] = fget_partial_pressure(
                self.concentration[i],
                self.molar_mass[i],
                temperature,
            )

    def get_molar_mass(self):
        return (self.molar_mass[0] if self.n_species == 1
                else self.molar_mass.to_numpy())

    def get_concentration(self):
        return (self.concentration[0] if self.n_species == 1
                else self.concentration.to_numpy())

    def get_pure_vapor_pressure(self, temperature):
        # delegate to stored strategy objects (scalar python loop)
        if self.n_species == 1:
            return self.strategies[0].pure_vapor_pressure(temperature)
        return np.array(
            [s.pure_vapor_pressure(temperature) for s in self.strategies],
            dtype=np.float64,
        )

    def get_partial_pressure(self, temperature):
        buffer = np.empty(self.n_species, dtype=np.float64)
        self._partial_pressure_kernel(float(temperature), buffer)
        return buffer[0] if self.n_species == 1 else buffer

    def get_saturation_ratio(self, temperature):
        pp   = self.get_partial_pressure(temperature)
        p_vp = self.get_pure_vapor_pressure(temperature)
        return pp / p_vp

    def get_saturation_concentration(self, temperature):
        return np.array([
            s.saturation_concentration(
                molar_mass=self.molar_mass[i],
                temperature=temperature,
            )
            for i, s in enumerate(self.strategies)
        ], dtype=np.float64)[0] if self.n_species == 1 else np.array([
            s.saturation_concentration(
                molar_mass=self.molar_mass[i],
                temperature=temperature,
            ) for i, s in enumerate(self.strategies)
        ], dtype=np.float64)

    # same signature as NumPy class
    def add_concentration(self, delta):
        delta = np.atleast_1d(np.asarray(delta, dtype=np.float64))
        if delta.size == 1 and self.n_species > 1:
            delta = np.full(self.n_species, delta.item(), dtype=np.float64)
        if delta.size != self.n_species:
            raise ValueError("delta length mismatch")
        for i in range(self.n_species):
            self.concentration[i] = max(
                0.0, self.concentration[i] + delta[i]
            )

    def set_concentration(self, new_value):
        self.add_concentration(np.asarray(new_value, dtype=np.float64)
                               - self.get_concentration())

    # meta dunders (str / len / + / +=) can remain python-side only

@register("GasSpecies", backend="taichi")
def ti_create_gas_species(*args, **kwargs):     # noqa: D401
    """Factory so external code can request GasSpecies via backend."""
    return GasSpecies(*args, **kwargs)

__all__ = ["GasSpecies", "ti_create_gas_species"]
