"""Taichi-accelerated aerodynamic length calculation module."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

@ti.func
def fget_aerodynamic_length(
    physical_length: ti.f64,
    physical_slip: ti.f64,
    aerodynamic_slip: ti.f64,
    density: ti.f64,
    reference_density: ti.f64,
    shape_factor: ti.f64
) -> ti.f64:
    """Element-wise Taichi function for aerodynamic length."""
    return physical_length * ti.sqrt(
        (physical_slip / aerodynamic_slip) *
        (density / (reference_density * shape_factor))
    )

@ti.kernel
def kget_aerodynamic_length(
    pl: ti.types.ndarray(dtype=ti.f64, ndim=1),
    psc: ti.types.ndarray(dtype=ti.f64, ndim=1),
    asc: ti.types.ndarray(dtype=ti.f64, ndim=1),
    rho: ti.types.ndarray(dtype=ti.f64, ndim=1),
    reference_density: ti.f64,
    shape_factor: ti.f64,
    result: ti.types.ndarray(dtype=ti.f64, ndim=1)
):
    """Vectorized Taichi kernel for aerodynamic length."""
    for i in range(result.shape[0]):
        result[i] = fget_aerodynamic_length(
            pl[i], psc[i], asc[i], rho[i],
            reference_density, shape_factor
        )

@register("get_aerodynamic_length", backend="taichi")
def ti_get_aerodynamic_length(
    pl,
    psc,
    asc,
    rho,
    reference_density: float = 1000.0,
    aerodynamic_shape_factor: float = 1.0
):
    """Taichi wrapper for aerodynamic length calculation."""
    # 5 a – type guard
    if not (
        isinstance(pl, np.ndarray)
        and isinstance(psc, np.ndarray)
        and isinstance(asc, np.ndarray)
        and isinstance(rho, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for all inputs.")

    # 5 b – ensure 1-D float64 NumPy arrays
    pl = np.atleast_1d(pl).astype(np.float64)
    psc = np.atleast_1d(psc).astype(np.float64)
    asc = np.atleast_1d(asc).astype(np.float64)
    rho = np.atleast_1d(rho).astype(np.float64)
    n = pl.size

    # 5 c – allocate Taichi NDArray buffers
    pl_ti = ti.ndarray(dtype=ti.f64, shape=n)
    psc_ti = ti.ndarray(dtype=ti.f64, shape=n)
    asc_ti = ti.ndarray(dtype=ti.f64, shape=n)
    rho_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    pl_ti.from_numpy(pl)
    psc_ti.from_numpy(psc)
    asc_ti.from_numpy(asc)
    rho_ti.from_numpy(rho)

    # 5 d – launch the kernel
    kget_aerodynamic_length(
        pl_ti, psc_ti, asc_ti, rho_ti,
        float(reference_density),
        float(aerodynamic_shape_factor),
        result_ti
    )

    # 5 e – convert result back to NumPy and unwrap if it is a single value
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np
