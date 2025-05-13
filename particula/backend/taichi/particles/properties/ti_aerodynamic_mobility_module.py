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
    particle_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    slip_correction_factor: ti.types.ndarray(dtype=ti.f64, ndim=1),
    dynamic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Vectorized Taichi kernel for aerodynamic mobility."""
    for i in range(result.shape[0]):
        result[i] = fget_aerodynamic_mobility(
            particle_radius[i],
            slip_correction_factor[i],
            dynamic_viscosity[i]
        )

@register("get_aerodynamic_mobility", backend="taichi")
def ti_get_aerodynamic_mobility(
    particle_radius,
    slip_correction_factor,
    dynamic_viscosity,
):
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
    pr_np  = np.asarray(particle_radius, dtype=np.float64)
    scf_np = np.asarray(slip_correction_factor, dtype=np.float64)
    dv_np  = np.asarray(dynamic_viscosity, dtype=np.float64)
    # if not (dv_np.shape == pr_np.shape):
    #     dv_np = np.ones_like(pr_np) * dv_np
    pr_b, scf_b, dv_b = np.broadcast_arrays(pr_np, scf_np, dv_np)

    flat_pr  = pr_b.ravel()
    flat_scf = scf_b.ravel()
    flat_dv  = dv_b.ravel()
    n = flat_pr.size

    # --- Taichi buffers -----------------------------------------------------
    pr_ti  = ti.ndarray(dtype=ti.f64, shape=n)
    scf_ti = ti.ndarray(dtype=ti.f64, shape=n)
    dv_ti  = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    pr_ti.from_numpy(flat_pr)
    scf_ti.from_numpy(flat_scf)
    dv_ti.from_numpy(flat_dv)

    # --- kernel -------------------------------------------------------------
    kget_aerodynamic_mobility(pr_ti, scf_ti, dv_ti, res_ti)

    # --- reshape / unwrap ---------------------------------------------------
    result_np = res_ti.to_numpy().reshape(pr_b.shape)
    return result_np.item() if result_np.size == 1 else result_np
