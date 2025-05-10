"""Taichi-accelerated implementation of get_partial_pressure_delta."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

@ti.func
def fget_partial_pressure_delta(
    partial_pressure_gas: ti.f64,
    partial_pressure_particle: ti.f64,
    kelvin_term: ti.f64,
) -> ti.f64:
    return partial_pressure_gas - partial_pressure_particle * kelvin_term


@ti.kernel
def kget_partial_pressure_delta(
    partial_pressure_gas: ti.types.ndarray(dtype=ti.f64, ndim=1),
    partial_pressure_particle: ti.types.ndarray(dtype=ti.f64, ndim=1),
    kelvin_term: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_partial_pressure_delta(
            partial_pressure_gas[i], partial_pressure_particle[i], kelvin_term[i]
        )


@register("get_partial_pressure_delta", backend="taichi")
def ti_get_partial_pressure_delta(
    partial_pressure_gas,
    partial_pressure_particle,
    kelvin_term,
):
    if not (
        isinstance(partial_pressure_gas, np.ndarray)
        and isinstance(partial_pressure_particle, np.ndarray)
        and isinstance(kelvin_term, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for all inputs.")

    pg, pp, kt = map(np.atleast_1d, (partial_pressure_gas, partial_pressure_particle, kelvin_term))
    if not (pg.shape == pp.shape == kt.shape):
        raise ValueError("All input arrays must share the same shape.")
    n = pg.size

    pg_ti = ti.ndarray(dtype=ti.f64, shape=n)
    pp_ti = ti.ndarray(dtype=ti.f64, shape=n)
    kt_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    pg_ti.from_numpy(pg)
    pp_ti.from_numpy(pp)
    kt_ti.from_numpy(kt)

    # launch the kernel
    kget_partial_pressure_delta(pg_ti, pp_ti, kt_ti, res_ti)

    # convert back to NumPy and unwrap scalar
    result_np = res_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np
