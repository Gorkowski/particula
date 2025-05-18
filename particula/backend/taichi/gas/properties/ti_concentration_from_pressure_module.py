"""Taichi version of get_concentration_from_pressure."""
import taichi as ti
import numpy as np

from particula.backend.dispatch_register import register
from particula.util.constants import GAS_CONSTANT

_GAS_CONSTANT = float(GAS_CONSTANT)

@ti.func
def fget_concentration_from_pressure(
    partial_pressure: ti.f64,
    molar_mass: ti.f64,
    temperature: ti.f64,
) -> ti.f64:
    return (partial_pressure * molar_mass) / (_GAS_CONSTANT * temperature)

@ti.kernel
def kget_concentration_from_pressure(
    partial_pressure: ti.types.ndarray(dtype=ti.f64, ndim=1),
    molar_mass: ti.types.ndarray(dtype=ti.f64, ndim=1),
    temperature: ti.types.ndarray(dtype=ti.f64, ndim=1),
    concentration: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(concentration.shape[0]):
        concentration[i] = fget_concentration_from_pressure(
            partial_pressure[i], molar_mass[i], temperature[i]
        )

@register("get_concentration_from_pressure", backend="taichi")
def ti_get_concentration_from_pressure(partial_pressure, molar_mass, temperature):
    """Taichi wrapper for get_concentration_from_pressure (vectorised)."""
    # 5 a – type guard  (explicit float64)
    partial_pressure = np.asarray(partial_pressure, dtype=np.float64)
    molar_mass       = np.asarray(molar_mass,       dtype=np.float64)
    temperature      = np.asarray(temperature,      dtype=np.float64)

    # 5 b – broadcast to common shape, then flatten
    partial_pressure_broadcast, molar_mass_broadcast, temperature_broadcast = (
        np.broadcast_arrays(partial_pressure, molar_mass, temperature)
    )
    flat_partial_pressure, flat_molar_mass, flat_temperature = map(
        np.ravel,
        (
            partial_pressure_broadcast,
            molar_mass_broadcast,
            temperature_broadcast,
        ),
    )
    n_elements = flat_partial_pressure.size

    # 5 c – allocate buffers with explicit names
    partial_pressure_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    molar_mass_ti       = ti.ndarray(dtype=ti.f64, shape=n_elements)
    temperature_ti      = ti.ndarray(dtype=ti.f64, shape=n_elements)
    concentration_ti    = ti.ndarray(dtype=ti.f64, shape=n_elements)
    partial_pressure_ti.from_numpy(flat_partial_pressure)
    molar_mass_ti.from_numpy(flat_molar_mass)
    temperature_ti.from_numpy(flat_temperature)

    # 5 d – launch kernel with explicit buffer names
    kget_concentration_from_pressure(
        partial_pressure_ti,
        molar_mass_ti,
        temperature_ti,
        concentration_ti,
    )

    # 5 e – return NumPy/scalar, restoring broadcasted shape
    concentration_np = concentration_ti.to_numpy().reshape(
        partial_pressure_broadcast.shape
    )
    return concentration_np.item() if concentration_np.size == 1 else concentration_np
