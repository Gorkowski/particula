"""Taichi implementation of lognormal PDF/PMF particle-size distributions."""
import taichi as ti
import numpy as np

from particula.backend.dispatch_register import register
from particula.particles.properties.convert_size_distribution import (
    get_pdf_distribution_in_pmf,
)

# --------------------------------------------------------------------------- #
# 1. Element-wise scalar Taichi function
# --------------------------------------------------------------------------- #
@ti.func
def fget_lognormal_pdf(x: ti.f64, mode: ti.f64, gsd: ti.f64) -> ti.f64:
    """Log-normal PDF for one x and one mode (all float64)."""
    sigma = ti.log(gsd)
    coeff = 0.0
    if x > 0.0:
        coeff = 1.0 / (x * sigma * ti.sqrt(2.0 * ti.math.pi))
    exponent = -((ti.log(x) - ti.log(mode)) ** 2) / (2.0 * sigma * sigma)
    return coeff * ti.exp(exponent)

# --------------------------------------------------------------------------- #
# 2. Vectorised kernel
# --------------------------------------------------------------------------- #
@ti.kernel
def kget_lognormal_pdf_distribution(                          # pragma: no cover
    x_vals: ti.types.ndarray(dtype=ti.f64, ndim=1),           # n_x
    modes: ti.types.ndarray(dtype=ti.f64, ndim=1),            # n_m
    gsds:  ti.types.ndarray(dtype=ti.f64, ndim=1),            # n_m
    weights: ti.types.ndarray(dtype=ti.f64, ndim=1),          # n_m
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),           # n_x
):
    """Compute Σ_j w_j · PDF_j(x_i) for every x_i."""
    for i in range(result.shape[0]):        # iterate over x
        acc = 0.0
        for j in range(modes.shape[0]):     # iterate over modes
            acc += weights[j] * fget_lognormal_pdf(
                x_vals[i], modes[j], gsds[j]
            )
        result[i] = acc

# --------------------------------------------------------------------------- #
# 3. NumPy helper to obtain per-mode weights  (number / area)
# --------------------------------------------------------------------------- #
def _compute_mode_weights(
    x_vals: np.ndarray,
    mode: np.ndarray,
    gsd: np.ndarray,
    n_part: np.ndarray,
) -> np.ndarray:
    """Return (number_of_particles / area) for every mode."""
    sigma = np.log(gsd)
    dist = np.exp(
        -((np.log(x_vals[:, None]) - np.log(mode)) ** 2) / (2.0 * sigma**2)
    ) / (x_vals[:, None] * sigma * np.sqrt(2.0 * np.pi))
    area = np.trapezoid(dist, x=x_vals[:, None], axis=0)
    area[area == 0] = np.nan
    return n_part / area

# --------------------------------------------------------------------------- #
# 4. Public Taichi wrappers with dispatch registration
# --------------------------------------------------------------------------- #
@register("get_lognormal_pdf_distribution", backend="taichi")
def ti_get_lognormal_pdf_distribution(
    x_values,
    mode,
    geometric_standard_deviation,
    number_of_particles,
):
    """Taichi-accelerated lognormal PDF distribution."""
    # 5 a – type guard
    for arr in (
        x_values,
        mode,
        geometric_standard_deviation,
        number_of_particles,
    ):
        if not isinstance(arr, np.ndarray):
            raise TypeError("Taichi backend expects NumPy arrays.")

    # 5 b – ensure 1-D NumPy arrays
    x_vals = np.atleast_1d(x_values)
    modes = np.atleast_1d(mode)
    gsds = np.atleast_1d(geometric_standard_deviation)
    n_parts = np.atleast_1d(number_of_particles)

    if not (x_vals.ndim == 1 and modes.shape == gsds.shape == n_parts.shape):
        raise ValueError(
            "mode, geometric_standard_deviation, and number_of_particles "
            "must all be 1-D and have identical shapes."
        )

    # 5 c – allocate Taichi buffers
    n_x, n_m = x_vals.size, modes.size
    x_ti = ti.ndarray(dtype=ti.f64, shape=n_x)
    mode_ti = ti.ndarray(dtype=ti.f64, shape=n_m)
    gsd_ti = ti.ndarray(dtype=ti.f64, shape=n_m)
    w_ti = ti.ndarray(dtype=ti.f64, shape=n_m)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n_x)

    x_ti.from_numpy(x_vals)
    mode_ti.from_numpy(modes)
    gsd_ti.from_numpy(gsds)
    w_ti.from_numpy(_compute_mode_weights(x_vals, modes, gsds, n_parts))

    # 5 d – launch kernel
    kget_lognormal_pdf_distribution(x_ti, mode_ti, gsd_ti, w_ti, result_ti)

    # 5 e – back to NumPy
    res = result_ti.to_numpy()
    return res.item() if res.size == 1 else res


@register("get_lognormal_pmf_distribution", backend="taichi")
def ti_get_lognormal_pmf_distribution(
    x_values,
    mode,
    geometric_standard_deviation,
    number_of_particles,
):
    """Taichi-accelerated lognormal PMF distribution."""
    pdf = ti_get_lognormal_pdf_distribution(
        x_values=x_values,
        mode=mode,
        geometric_standard_deviation=geometric_standard_deviation,
        number_of_particles=number_of_particles,
    )

    pmf = get_pdf_distribution_in_pmf(
        x_array=x_values, distribution=pdf, to_pdf=False
    )
    total = pmf.sum()
    return pmf * np.divide(
        number_of_particles.sum(),
        total,
        out=np.ones_like(total),
        where=total != 0.0,
    )
