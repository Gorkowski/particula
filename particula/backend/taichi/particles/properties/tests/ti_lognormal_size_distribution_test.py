"""Tests for the Taichi lognormal size-distribution functions."""
import numpy as np
import numpy.testing as npt
import taichi as ti

ti.init(arch=ti.cpu)

# Reference (NumPy/SciPy) implementation
from particula.particles.properties.lognormal_size_distribution import (
    get_lognormal_pdf_distribution as ref_pdf,
    get_lognormal_pmf_distribution as ref_pmf,
)

# Taichi implementation
from particula.backend.taichi.particles.properties.ti_lognormal_size_distribution_module import (  # noqa:E501
    ti_get_lognormal_pdf_distribution,
    ti_get_lognormal_pmf_distribution,
    kget_lognormal_pdf_distribution,
    _compute_mode_weights,
)


def _inputs():
    x_vals = np.linspace(1e-9, 1e-6, 100)
    mode = np.array([5e-8, 1e-7])
    gsd = np.array([1.5, 2.0])
    n_part = np.array([1e9, 5e9])
    return x_vals, mode, gsd, n_part


def test_pdf_wrapper():
    x_vals, mode, gsd, n_part = _inputs()
    npt.assert_allclose(
        ti_get_lognormal_pdf_distribution(x_vals, mode, gsd, n_part),
        ref_pdf(x_vals, mode, gsd, n_part),
        rtol=1e-7,
    )


def test_pmf_wrapper():
    x_vals, mode, gsd, n_part = _inputs()
    npt.assert_allclose(
        ti_get_lognormal_pmf_distribution(x_vals, mode, gsd, n_part),
        ref_pmf(x_vals, mode, gsd, n_part),
        rtol=1e-7,
    )


def test_kernel_direct():
    x_vals, mode, gsd, n_part = _inputs()
    weights = _compute_mode_weights(x_vals, mode, gsd, n_part)

    # Allocate Taichi buffers
    n_x, n_m = x_vals.size, mode.size
    x_ti = ti.ndarray(dtype=ti.f64, shape=n_x)
    mode_ti = ti.ndarray(dtype=ti.f64, shape=n_m)
    gsd_ti = ti.ndarray(dtype=ti.f64, shape=n_m)
    w_ti = ti.ndarray(dtype=ti.f64, shape=n_m)
    out_ti = ti.ndarray(dtype=ti.f64, shape=n_x)

    x_ti.from_numpy(x_vals)
    mode_ti.from_numpy(mode)
    gsd_ti.from_numpy(gsd)
    w_ti.from_numpy(weights)

    kget_lognormal_pdf_distribution(x_ti, mode_ti, gsd_ti, w_ti, out_ti)

    npt.assert_allclose(
        out_ti.to_numpy(),
        ref_pdf(x_vals, mode, gsd, n_part),
        rtol=1e-7,
    )
