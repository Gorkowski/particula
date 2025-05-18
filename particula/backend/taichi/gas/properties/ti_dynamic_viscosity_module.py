"""Taichi implementation of Sutherland’s dynamic-viscosity formula."""
import taichi as ti
import numpy as np

from particula.backend.dispatch_register import register
from particula.util.constants import (
    REF_TEMPERATURE_STP,
    REF_VISCOSITY_AIR_STP,
    SUTHERLAND_CONSTANT,
)

# ── 3. element-wise function ──────────────────────────────────────────
@ti.func
def fget_dynamic_viscosity(
    temperature: ti.f64,
    reference_viscosity: ti.f64,
    reference_temperature: ti.f64,
) -> ti.f64:
    return (
        reference_viscosity
        * (temperature / reference_temperature) ** 1.5
        * (reference_temperature + SUTHERLAND_CONSTANT)
        / (temperature + SUTHERLAND_CONSTANT)
    )

# ── 4. vectorised kernel ──────────────────────────────────────────────
@ti.kernel
def kget_dynamic_viscosity(                     # 1-D only
    temperature: ti.types.ndarray(dtype=ti.f64, ndim=1),
    reference_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    reference_temperature: ti.types.ndarray(dtype=ti.f64, ndim=1),
    dynamic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(dynamic_viscosity.shape[0]):
        dynamic_viscosity[i] = fget_dynamic_viscosity(
            temperature[i],
            reference_viscosity[i],
            reference_temperature[i],
        )

# ── 5. public wrapper, backend registration ───────────────────────────
@register("get_dynamic_viscosity", backend="taichi")
def ti_get_dynamic_viscosity(
    temperature,
    reference_viscosity: float = REF_VISCOSITY_AIR_STP,
    reference_temperature: float = REF_TEMPERATURE_STP,
):
    # 5 a – type / shape guards
    temperature_array = np.atleast_1d(temperature).astype(np.float64)

    reference_viscosity_array = (
        np.full_like(temperature_array, reference_viscosity, dtype=np.float64)
        if np.isscalar(reference_viscosity)
        else np.asarray(reference_viscosity, dtype=np.float64)
    )
    reference_temperature_array = (
        np.full_like(
            temperature_array,
            reference_temperature,
            dtype=np.float64,
        )
        if np.isscalar(reference_temperature)
        else np.asarray(reference_temperature, dtype=np.float64)
    )

    # make sure all inputs share the same shape
    (
        temperature_array,
        reference_viscosity_array,
        reference_temperature_array,
    ) = np.broadcast_arrays(
        temperature_array,
        reference_viscosity_array,
        reference_temperature_array,
    )
    n_elements = temperature_array.size

    # 5 c – allocate Taichi NDArrays
    temperature_field = ti.ndarray(dtype=ti.f64, shape=n_elements)
    reference_viscosity_field = ti.ndarray(dtype=ti.f64, shape=n_elements)
    reference_temperature_field = ti.ndarray(dtype=ti.f64, shape=n_elements)
    dynamic_viscosity_field = ti.ndarray(dtype=ti.f64, shape=n_elements)

    temperature_field.from_numpy(temperature_array)
    reference_viscosity_field.from_numpy(reference_viscosity_array)
    reference_temperature_field.from_numpy(reference_temperature_array)

    # 5 d – launch kernel
    kget_dynamic_viscosity(
        temperature_field,
        reference_viscosity_field,
        reference_temperature_field,
        dynamic_viscosity_field,
    )

    # 5 e – back to NumPy, squeeze scalar
    dynamic_viscosity_array = dynamic_viscosity_field.to_numpy()
    return (
        dynamic_viscosity_array.item()
        if dynamic_viscosity_array.size == 1
        else dynamic_viscosity_array
    )
