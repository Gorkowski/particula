"""Taichi implementation of aerodynamic mobility of a particle in a fluid."""
import taichi as ti
import numpy as np
from numbers import Number
from particula.backend.dispatch_register import register

@ti.func
def fget_aerodynamic_mobility(
    particle_radius: ti.f64,
    slip_correction_factor: ti.f64,
    dynamic_viscosity: ti.f64,
) -> ti.f64:
    """Elementwise Taichi version of aerodynamic mobility."""
    return slip_correction_factor / (6.0 * ti.math.pi * dynamic_viscosity * particle_radius)

@ti.kernel
def kget_aerodynamic_mobility(
    particle_radius_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
    slip_correction_factor_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
    dynamic_viscosity_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Vectorized Taichi kernel for aerodynamic mobility."""
    for i in range(result_array.shape[0]):
        result_array[i] = fget_aerodynamic_mobility(
            particle_radius_array[i],
            slip_correction_factor_array[i],
            dynamic_viscosity_array[i]
        )

@register("get_aerodynamic_mobility", backend="taichi")
def ti_get_aerodynamic_mobility(
    particle_radius,
    slip_correction_factor,
    dynamic_viscosity,
):
    """Taichi backend wrapper for get_aerodynamic_mobility (broadcasts & dispatches)."""
    # --- type guard ---------------------------------------------------------
    if not (
        isinstance(particle_radius, (np.ndarray, Number))
        and isinstance(slip_correction_factor, (np.ndarray, Number))
        and isinstance(dynamic_viscosity, (np.ndarray, Number))
    ):
        raise TypeError(
            "Taichi backend expects NumPy arrays or scalars for all inputs."
        )
    # --- broadcast ----------------------------------------------------------
    particle_radius_np = np.asarray(particle_radius, dtype=np.float64)
    slip_correction_factor_np = np.asarray(slip_correction_factor, dtype=np.float64)
    dynamic_viscosity_np = np.asarray(dynamic_viscosity, dtype=np.float64)
    particle_radius_b, slip_correction_factor_b, dynamic_viscosity_b = np.broadcast_arrays(
        particle_radius_np, slip_correction_factor_np, dynamic_viscosity_np
    )

    particle_radius_flat = particle_radius_b.ravel()
    slip_correction_factor_flat = slip_correction_factor_b.ravel()
    dynamic_viscosity_flat = dynamic_viscosity_b.ravel()
    n_elements = particle_radius_flat.size

    # --- Taichi buffers -----------------------------------------------------
    particle_radius_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    slip_correction_factor_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    dynamic_viscosity_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    particle_radius_ti.from_numpy(particle_radius_flat)
    slip_correction_factor_ti.from_numpy(slip_correction_factor_flat)
    dynamic_viscosity_ti.from_numpy(dynamic_viscosity_flat)

    # --- kernel -------------------------------------------------------------
    kget_aerodynamic_mobility(
        particle_radius_ti,
        slip_correction_factor_ti,
        dynamic_viscosity_ti,
        result_ti
    )

    # --- reshape / unwrap ---------------------------------------------------
    result_np = result_ti.to_numpy().reshape(particle_radius_b.shape)
    return result_np.item() if result_np.size == 1 else result_np
