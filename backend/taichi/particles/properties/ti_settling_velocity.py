"""Taichi-accelerated particle settling velocity in a fluid."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

@ti.func
def fget_settling_velocity(
    particle_radius: ti.f64,
    particle_density: ti.f64,
    slip_correction_factor: ti.f64,
    dynamic_viscosity: ti.f64,
    gravitational_acceleration: ti.f64,
    fluid_density: ti.f64,
) -> ti.f64:
    """Elementwise Stokes settling velocity with slip correction."""
    return (
        ((2.0 * particle_radius) ** 2)
        * (particle_density - fluid_density)
        * slip_correction_factor
        * gravitational_acceleration
        / (18.0 * dynamic_viscosity)
    )

@ti.kernel
def kget_settling_velocity(
    particle_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    particle_density: ti.types.ndarray(dtype=ti.f64, ndim=1),
    slip_correction_factor: ti.types.ndarray(dtype=ti.f64, ndim=1),
    dynamic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    gravitational_acceleration: ti.types.ndarray(dtype=ti.f64, ndim=1),
    fluid_density: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Vectorized Taichi kernel for settling velocity."""
    for i in range(result.shape[0]):
        result[i] = fget_settling_velocity(
            particle_radius[i],
            particle_density[i],
            slip_correction_factor[i],
            dynamic_viscosity[i],
            gravitational_acceleration[i],
            fluid_density[i],
        )

@register("get_particle_settling_velocity", backend="taichi")
def ti_get_particle_settling_velocity(
    particle_radius,
    particle_density,
    slip_correction_factor,
    dynamic_viscosity,
    gravitational_acceleration=9.80665,
    fluid_density=0.0,
):
    """Taichi wrapper for particle settling velocity."""
    if not (
        isinstance(particle_radius, np.ndarray)
        and isinstance(particle_density, np.ndarray)
        and isinstance(slip_correction_factor, np.ndarray)
        and isinstance(dynamic_viscosity, (np.ndarray, float))
    ):
        raise TypeError("Taichi backend expects NumPy arrays for all inputs except dynamic_viscosity, which may be float or array.")

    # Ensure all inputs are 1D arrays of the same shape
    a1 = np.atleast_1d(particle_radius)
    a2 = np.atleast_1d(particle_density)
    a3 = np.atleast_1d(slip_correction_factor)
    a4 = np.atleast_1d(dynamic_viscosity)
    a5 = np.atleast_1d(gravitational_acceleration)
    a6 = np.atleast_1d(fluid_density)
    n = np.broadcast(a1, a2, a3, a4, a5, a6)[0].size

    # Broadcast all arrays to the same shape
    a1, a2, a3, a4, a5, a6 = np.broadcast_arrays(a1, a2, a3, a4, a5, a6)

    # Allocate Taichi NDArray buffers
    a1_ti = ti.ndarray(dtype=ti.f64, shape=n)
    a2_ti = ti.ndarray(dtype=ti.f64, shape=n)
    a3_ti = ti.ndarray(dtype=ti.f64, shape=n)
    a4_ti = ti.ndarray(dtype=ti.f64, shape=n)
    a5_ti = ti.ndarray(dtype=ti.f64, shape=n)
    a6_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    a1_ti.from_numpy(a1)
    a2_ti.from_numpy(a2)
    a3_ti.from_numpy(a3)
    a4_ti.from_numpy(a4)
    a5_ti.from_numpy(a5)
    a6_ti.from_numpy(a6)

    # Launch the kernel
    kget_settling_velocity(
        a1_ti, a2_ti, a3_ti, a4_ti, a5_ti, a6_ti, result_ti
    )

    # Convert result back to NumPy and unwrap if it is a single value
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np
