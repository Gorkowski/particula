"""Taichi-accelerated lognormal size distribution properties."""

import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

@ti.func
def fget_lognormal_pdf_distribution(
    x_value: ti.f64,
    mode: ti.f64,
    geometric_standard_deviation: ti.f64,
) -> ti.f64:
    """Element-wise lognormal PDF for a single mode."""
    # Implements the lognormal PDF formula
    # PDF(x) = (1 / [x * ln(gsd) * sqrt(2*pi)]) * exp(- (ln(x) - ln(mode))^2 / (2 * (ln(gsd))^2))
    sqrt_2pi = 2.5066282746310002  # sqrt(2*pi)
    if x_value <= 0.0 or mode <= 0.0 or geometric_standard_deviation <= 0.0:
        return 0.0
    ln_gsd = ti.log(geometric_standard_deviation)
    ln_x = ti.log(x_value)
    ln_mode = ti.log(mode)
    exponent = -((ln_x - ln_mode) * (ln_x - ln_mode)) / (2.0 * ln_gsd * ln_gsd)
    denominator = x_value * ln_gsd * sqrt_2pi
    return ti.exp(exponent) / denominator

@ti.kernel
def kget_lognormal_pdf_distribution(
    x_values: ti.types.ndarray(dtype=ti.f64, ndim=1),
    mode: ti.types.ndarray(dtype=ti.f64, ndim=1),
    geometric_standard_deviation: ti.types.ndarray(dtype=ti.f64, ndim=1),
    number_of_particles: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Vectorized lognormal PDF for multiple modes, summed over all modes."""
    n_x = result.shape[0]
    n_modes = mode.shape[0]
    for i in range(n_x):
        total = 0.0
        for j in range(n_modes):
            pdf = fget_lognormal_pdf_distribution(
                x_values[i], mode[j], geometric_standard_deviation[j]
            )
            total += pdf * number_of_particles[j]
        result[i] = total

@register("get_lognormal_pdf_distribution", backend="taichi")
def ti_get_lognormal_pdf_distribution(
    x_values, mode, geometric_standard_deviation, number_of_particles
):
    """Taichi-accelerated lognormal PDF distribution for multiple modes."""
    # Type guard
    if not (
        isinstance(x_values, np.ndarray)
        and isinstance(mode, np.ndarray)
        and isinstance(geometric_standard_deviation, np.ndarray)
        and isinstance(number_of_particles, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for all inputs.")

    # Ensure 1-D NumPy arrays
    x_values = np.atleast_1d(x_values)
    mode = np.atleast_1d(mode)
    geometric_standard_deviation = np.atleast_1d(geometric_standard_deviation)
    number_of_particles = np.atleast_1d(number_of_particles)
    n_x = x_values.size
    n_modes = mode.size

    # Allocate Taichi NDArray buffers
    x_values_ti = ti.ndarray(dtype=ti.f64, shape=n_x)
    mode_ti = ti.ndarray(dtype=ti.f64, shape=n_modes)
    gsd_ti = ti.ndarray(dtype=ti.f64, shape=n_modes)
    n_particles_ti = ti.ndarray(dtype=ti.f64, shape=n_modes)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n_x)
    x_values_ti.from_numpy(x_values)
    mode_ti.from_numpy(mode)
    gsd_ti.from_numpy(geometric_standard_deviation)
    n_particles_ti.from_numpy(number_of_particles)

    # Launch the kernel
    kget_lognormal_pdf_distribution(
        x_values_ti, mode_ti, gsd_ti, n_particles_ti, result_ti
    )

    # Convert result back to NumPy and unwrap if it is a single value
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np
