import taichi as ti
ti.init(arch=ti.cpu)

import numpy as np
import numpy.testing as npt

from particula.particles.properties.convert_size_distribution import (
    get_distribution_in_dn as py_get_distribution_in_dn,
    get_pdf_distribution_in_pmf as py_get_pdf_distribution_in_pmf,
)

from particula.backend.taichi.particles.properties.ti_convert_size_distribution_module import (
    ti_get_distribution_in_dn,
    ti_get_pdf_distribution_in_pmf,
    kget_distribution_in_dn,
    kget_pdf_distribution_in_pmf,
)

def test_wrapper_distribution_in_dn():
    d = np.linspace(1e-8, 1e-6, 6)
    dn = np.linspace(1e5, 1e4, 6)
    npt.assert_allclose(
        ti_get_distribution_in_dn(d, dn, inverse=False),
        py_get_distribution_in_dn(d, dn, inverse=False),
        rtol=1e-12,
    )

def test_wrapper_pdf_pmf():
    x = np.linspace(1.0, 6.0, 6)
    pmf = np.linspace(10.0, 2.0, 6)
    npt.assert_allclose(
        ti_get_pdf_distribution_in_pmf(x, pmf, to_pdf=True),
        py_get_pdf_distribution_in_pmf(x, pmf, to_pdf=True),
        rtol=1e-12,
    )

def test_kernel_distribution_in_dn():
    d = np.linspace(1e-8, 1e-6, 6)
    dn = np.linspace(1e5, 1e4, 6)
    n = d.size
    d_ti = ti.ndarray(dtype=ti.f64, shape=n); d_ti.from_numpy(d)
    dn_ti = ti.ndarray(dtype=ti.f64, shape=n); dn_ti.from_numpy(dn)
    out_ti = ti.ndarray(dtype=ti.f64, shape=n)
    kget_distribution_in_dn(d_ti, dn_ti, 0, out_ti)
    npt.assert_allclose(out_ti.to_numpy(), py_get_distribution_in_dn(d, dn))

def test_kernel_pdf_pmf():
    x = np.linspace(1.0, 6.0, 6)
    pmf = np.linspace(10.0, 2.0, 6)
    n = x.size
    x_ti = ti.ndarray(dtype=ti.f64, shape=n); x_ti.from_numpy(x)
    pmf_ti = ti.ndarray(dtype=ti.f64, shape=n); pmf_ti.from_numpy(pmf)
    out_ti = ti.ndarray(dtype=ti.f64, shape=n)
    kget_pdf_distribution_in_pmf(x_ti, pmf_ti, 1, out_ti)
    npt.assert_allclose(
        out_ti.to_numpy(),
        py_get_pdf_distribution_in_pmf(x, pmf, to_pdf=True),
    )
