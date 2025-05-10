import numpy as np
import taichi as ti
ti.init(arch=ti.cpu)

from particula.backend.taichi.particles.properties.ti_partial_pressure_module import (
    kget_partial_pressure_delta,
    ti_get_partial_pressure_delta,
)

def reference(pg, pp, kt):
    return pg - pp * kt

def test_wrapper_matches_numpy():
    pg = np.array([1000.0, 950.0])
    pp = np.array([900.0, 850.0])
    kt = np.array([1.01, 1.02])
    np.testing.assert_allclose(
        ti_get_partial_pressure_delta(pg, pp, kt),
        reference(pg, pp, kt),
    )

def test_kernel_direct():
    pg = np.array([1000.0, 950.0])
    pp = np.array([900.0, 850.0])
    kt = np.array([1.01, 1.02])
    n = pg.size
    pg_ti = ti.ndarray(dtype=ti.f64, shape=n)
    pp_ti = ti.ndarray(dtype=ti.f64, shape=n)
    kt_ti = ti.ndarray(dtype=ti.f64, shape=n)
    out_ti = ti.ndarray(dtype=ti.f64, shape=n)
    pg_ti.from_numpy(pg)
    pp_ti.from_numpy(pp)
    kt_ti.from_numpy(kt)
    kget_partial_pressure_delta(pg_ti, pp_ti, kt_ti, out_ti)
    np.testing.assert_allclose(out_ti.to_numpy(), reference(pg, pp, kt))
