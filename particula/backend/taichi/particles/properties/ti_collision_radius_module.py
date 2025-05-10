"""Taichi-accelerated collision radius models."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

@ti.func
def fget_collision_radius_mg1988(gyration_radius: ti.f64) -> ti.f64:
    """Taichi version: R_c = R_g"""
    return gyration_radius

@ti.func
def fget_collision_radius_sr1992(gyration_radius: ti.f64, fractal_dimension: ti.f64) -> ti.f64:
    """Taichi version: R_c = sqrt((d_f + 2) / 3) * R_g"""
    return ti.sqrt((fractal_dimension + 2.0) / 3.0) * gyration_radius

@ti.func
def fget_collision_radius_mzg2002(gyration_radius: ti.f64, fractal_prefactor: ti.f64) -> ti.f64:
    """Taichi version: R_c = 1.037 * (k0^0.077) * R_g"""
    return 1.037 * ti.pow(fractal_prefactor, 0.077) * gyration_radius

@ti.func
def fget_collision_radius_tt2012(
    fractal_dimension: ti.f64,
    number_of_particles: ti.f64,
    gyration_radius: ti.f64,
    radius_monomer: ti.f64,
) -> ti.f64:
    """Taichi version: ported from NumPy."""
    alpha1 = 0.253 * fractal_dimension**2 - 1.209 * fractal_dimension + 1.433
    alpha2 = -0.218 * fractal_dimension**2 + 0.964 * fractal_dimension - 0.180
    phi = 1.0 / (alpha1 * ti.log(number_of_particles) + alpha2)
    radius_s_i = phi * gyration_radius
    radius_s_ii = (
        radius_monomer * (1.203 - 0.4315 / fractal_dimension) / 2.0
    ) * ti.pow(4.0 * radius_s_i / radius_monomer, 0.8806 + 0.3497 / fractal_dimension)
    return radius_s_ii / 2.0

@ti.func
def fget_collision_radius_wq2022_rg(gyration_radius: ti.f64, radius_monomer: ti.f64) -> ti.f64:
    """Taichi version: R_c = (0.973 * (R_g / r_m) + 0.441) * r_m"""
    return (0.973 * (gyration_radius / radius_monomer) + 0.441) * radius_monomer

@ti.func
def fget_collision_radius_wq2022_rg_df(
    fractal_dimension: ti.f64,
    gyration_radius: ti.f64,
    radius_monomer: ti.f64,
) -> ti.f64:
    """Taichi version: R_c = (0.882 * d_f^0.223 * (R_g / r_m) + 0.387) * r_m"""
    return (
        0.882 * ti.pow(fractal_dimension, 0.223) * (gyration_radius / radius_monomer) + 0.387
    ) * radius_monomer

@ti.func
def fget_collision_radius_wq2022_rg_df_k0(
    fractal_dimension: ti.f64,
    fractal_prefactor: ti.f64,
    gyration_radius: ti.f64,
    radius_monomer: ti.f64,
) -> ti.f64:
    """Taichi version: R_c = (0.777*d_f^0.479*k0^0.000970*(R_g/r_m)+0.267*k0-0.079)*r_m"""
    return (
        0.777 * ti.pow(fractal_dimension, 0.479)
        * ti.pow(fractal_prefactor, 0.000970)
        * (gyration_radius / radius_monomer)
        + 0.267 * fractal_prefactor
        - 0.079
    ) * radius_monomer

@ti.func
def fget_collision_radius_wq2022_rg_df_k0_a13(
    fractal_dimension: ti.f64,
    fractal_prefactor: ti.f64,
    shape_anisotropy: ti.f64,
    gyration_radius: ti.f64,
    radius_monomer: ti.f64,
) -> ti.f64:
    """Taichi version: (0.876*d_f^0.363*k0^-0.105*(R_g/r_m)+0.421*k0-0.036*a13-0.227)*r_m"""
    return (
        0.876 * ti.pow(fractal_dimension, 0.363)
        * ti.pow(fractal_prefactor, -0.105)
        * (gyration_radius / radius_monomer)
        + 0.421 * fractal_prefactor
        - 0.036 * shape_anisotropy
        - 0.227
    ) * radius_monomer

@ti.kernel
def kget_collision_radius_mg1988(
    gyration_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    out: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Kernel for mg1988."""
    for i in range(out.shape[0]):
        out[i] = fget_collision_radius_mg1988(gyration_radius[i])

@ti.kernel
def kget_collision_radius_sr1992(
    gyration_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    fractal_dimension: ti.types.ndarray(dtype=ti.f64, ndim=1),
    out: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Kernel for sr1992."""
    for i in range(out.shape[0]):
        out[i] = fget_collision_radius_sr1992(gyration_radius[i], fractal_dimension[i])

@ti.kernel
def kget_collision_radius_mzg2002(
    gyration_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    fractal_prefactor: ti.types.ndarray(dtype=ti.f64, ndim=1),
    out: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Kernel for mzg2002."""
    for i in range(out.shape[0]):
        out[i] = fget_collision_radius_mzg2002(gyration_radius[i], fractal_prefactor[i])

@ti.kernel
def kget_collision_radius_tt2012(
    fractal_dimension: ti.types.ndarray(dtype=ti.f64, ndim=1),
    number_of_particles: ti.types.ndarray(dtype=ti.f64, ndim=1),
    gyration_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    radius_monomer: ti.types.ndarray(dtype=ti.f64, ndim=1),
    out: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Kernel for tt2012."""
    for i in range(out.shape[0]):
        out[i] = fget_collision_radius_tt2012(
            fractal_dimension[i],
            number_of_particles[i],
            gyration_radius[i],
            radius_monomer[i],
        )

@ti.kernel
def kget_collision_radius_wq2022_rg(
    gyration_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    radius_monomer: ti.types.ndarray(dtype=ti.f64, ndim=1),
    out: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Kernel for wq2022_rg."""
    for i in range(out.shape[0]):
        out[i] = fget_collision_radius_wq2022_rg(gyration_radius[i], radius_monomer[i])

@ti.kernel
def kget_collision_radius_wq2022_rg_df(
    fractal_dimension: ti.types.ndarray(dtype=ti.f64, ndim=1),
    gyration_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    radius_monomer: ti.types.ndarray(dtype=ti.f64, ndim=1),
    out: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Kernel for wq2022_rg_df."""
    for i in range(out.shape[0]):
        out[i] = fget_collision_radius_wq2022_rg_df(
            fractal_dimension[i], gyration_radius[i], radius_monomer[i]
        )

@ti.kernel
def kget_collision_radius_wq2022_rg_df_k0(
    fractal_dimension: ti.types.ndarray(dtype=ti.f64, ndim=1),
    fractal_prefactor: ti.types.ndarray(dtype=ti.f64, ndim=1),
    gyration_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    radius_monomer: ti.types.ndarray(dtype=ti.f64, ndim=1),
    out: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Kernel for wq2022_rg_df_k0."""
    for i in range(out.shape[0]):
        out[i] = fget_collision_radius_wq2022_rg_df_k0(
            fractal_dimension[i],
            fractal_prefactor[i],
            gyration_radius[i],
            radius_monomer[i],
        )

@ti.kernel
def kget_collision_radius_wq2022_rg_df_k0_a13(
    fractal_dimension: ti.types.ndarray(dtype=ti.f64, ndim=1),
    fractal_prefactor: ti.types.ndarray(dtype=ti.f64, ndim=1),
    shape_anisotropy: ti.types.ndarray(dtype=ti.f64, ndim=1),
    gyration_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    radius_monomer: ti.types.ndarray(dtype=ti.f64, ndim=1),
    out: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Kernel for wq2022_rg_df_k0_a13."""
    for i in range(out.shape[0]):
        out[i] = fget_collision_radius_wq2022_rg_df_k0_a13(
            fractal_dimension[i],
            fractal_prefactor[i],
            shape_anisotropy[i],
            gyration_radius[i],
            radius_monomer[i],
        )

def _ensure_all_ndarrays(*args):
    for arg in args:
        if not isinstance(arg, np.ndarray):
            raise TypeError("Taichi backend expects NumPy arrays for all inputs.")

def _ensure_all_same_size(*args):
    sizes = [np.atleast_1d(arg).size for arg in args]
    if any(sz != sizes[0] for sz in sizes):
        raise ValueError("All input arrays must have the same size.")

@register("get_collision_radius_mg1988", backend="taichi")
def ti_get_collision_radius_mg1988(gyration_radius):
    """Taichi wrapper for mg1988."""
    _ensure_all_ndarrays(gyration_radius)
    a1 = np.atleast_1d(gyration_radius)
    n = a1.size
    variable_a1_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a1_ti.from_numpy(a1)
    kget_collision_radius_mg1988(variable_a1_ti, result_ti)
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np

@register("get_collision_radius_sr1992", backend="taichi")
def ti_get_collision_radius_sr1992(gyration_radius, fractal_dimension):
    """Taichi wrapper for sr1992."""
    _ensure_all_ndarrays(gyration_radius, fractal_dimension)
    a1 = np.atleast_1d(gyration_radius)
    a2 = np.atleast_1d(fractal_dimension)
    _ensure_all_same_size(a1, a2)
    n = a1.size
    variable_a1_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a2_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a1_ti.from_numpy(a1)
    variable_a2_ti.from_numpy(a2)
    kget_collision_radius_sr1992(variable_a1_ti, variable_a2_ti, result_ti)
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np

@register("get_collision_radius_mzg2002", backend="taichi")
def ti_get_collision_radius_mzg2002(gyration_radius, fractal_prefactor):
    """Taichi wrapper for mzg2002."""
    _ensure_all_ndarrays(gyration_radius, fractal_prefactor)
    a1 = np.atleast_1d(gyration_radius)
    a2 = np.atleast_1d(fractal_prefactor)
    _ensure_all_same_size(a1, a2)
    n = a1.size
    variable_a1_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a2_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a1_ti.from_numpy(a1)
    variable_a2_ti.from_numpy(a2)
    kget_collision_radius_mzg2002(variable_a1_ti, variable_a2_ti, result_ti)
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np

@register("get_collision_radius_tt2012", backend="taichi")
def ti_get_collision_radius_tt2012(fractal_dimension, number_of_particles, gyration_radius, radius_monomer):
    """Taichi wrapper for tt2012."""
    _ensure_all_ndarrays(fractal_dimension, number_of_particles, gyration_radius, radius_monomer)
    a1 = np.atleast_1d(fractal_dimension)
    a2 = np.atleast_1d(number_of_particles)
    a3 = np.atleast_1d(gyration_radius)
    a4 = np.atleast_1d(radius_monomer)
    _ensure_all_same_size(a1, a2, a3, a4)
    n = a1.size
    variable_a1_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a2_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a3_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a4_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a1_ti.from_numpy(a1)
    variable_a2_ti.from_numpy(a2)
    variable_a3_ti.from_numpy(a3)
    variable_a4_ti.from_numpy(a4)
    kget_collision_radius_tt2012(variable_a1_ti, variable_a2_ti, variable_a3_ti, variable_a4_ti, result_ti)
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np

@register("get_collision_radius_wq2022_rg", backend="taichi")
def ti_get_collision_radius_wq2022_rg(gyration_radius, radius_monomer):
    """Taichi wrapper for wq2022_rg."""
    _ensure_all_ndarrays(gyration_radius, radius_monomer)
    a1 = np.atleast_1d(gyration_radius)
    a2 = np.atleast_1d(radius_monomer)
    _ensure_all_same_size(a1, a2)
    n = a1.size
    variable_a1_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a2_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a1_ti.from_numpy(a1)
    variable_a2_ti.from_numpy(a2)
    kget_collision_radius_wq2022_rg(variable_a1_ti, variable_a2_ti, result_ti)
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np

@register("get_collision_radius_wq2022_rg_df", backend="taichi")
def ti_get_collision_radius_wq2022_rg_df(fractal_dimension, gyration_radius, radius_monomer):
    """Taichi wrapper for wq2022_rg_df."""
    _ensure_all_ndarrays(fractal_dimension, gyration_radius, radius_monomer)
    a1 = np.atleast_1d(fractal_dimension)
    a2 = np.atleast_1d(gyration_radius)
    a3 = np.atleast_1d(radius_monomer)
    _ensure_all_same_size(a1, a2, a3)
    n = a1.size
    variable_a1_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a2_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a3_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a1_ti.from_numpy(a1)
    variable_a2_ti.from_numpy(a2)
    variable_a3_ti.from_numpy(a3)
    kget_collision_radius_wq2022_rg_df(variable_a1_ti, variable_a2_ti, variable_a3_ti, result_ti)
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np

@register("get_collision_radius_wq2022_rg_df_k0", backend="taichi")
def ti_get_collision_radius_wq2022_rg_df_k0(fractal_dimension, fractal_prefactor, gyration_radius, radius_monomer):
    """Taichi wrapper for wq2022_rg_df_k0."""
    _ensure_all_ndarrays(fractal_dimension, fractal_prefactor, gyration_radius, radius_monomer)
    a1 = np.atleast_1d(fractal_dimension)
    a2 = np.atleast_1d(fractal_prefactor)
    a3 = np.atleast_1d(gyration_radius)
    a4 = np.atleast_1d(radius_monomer)
    _ensure_all_same_size(a1, a2, a3, a4)
    n = a1.size
    variable_a1_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a2_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a3_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a4_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a1_ti.from_numpy(a1)
    variable_a2_ti.from_numpy(a2)
    variable_a3_ti.from_numpy(a3)
    variable_a4_ti.from_numpy(a4)
    kget_collision_radius_wq2022_rg_df_k0(variable_a1_ti, variable_a2_ti, variable_a3_ti, variable_a4_ti, result_ti)
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np

@register("get_collision_radius_wq2022_rg_df_k0_a13", backend="taichi")
def ti_get_collision_radius_wq2022_rg_df_k0_a13(fractal_dimension, fractal_prefactor, shape_anisotropy, gyration_radius, radius_monomer):
    """Taichi wrapper for wq2022_rg_df_k0_a13."""
    _ensure_all_ndarrays(fractal_dimension, fractal_prefactor, shape_anisotropy, gyration_radius, radius_monomer)
    a1 = np.atleast_1d(fractal_dimension)
    a2 = np.atleast_1d(fractal_prefactor)
    a3 = np.atleast_1d(shape_anisotropy)
    a4 = np.atleast_1d(gyration_radius)
    a5 = np.atleast_1d(radius_monomer)
    _ensure_all_same_size(a1, a2, a3, a4, a5)
    n = a1.size
    variable_a1_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a2_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a3_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a4_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a5_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a1_ti.from_numpy(a1)
    variable_a2_ti.from_numpy(a2)
    variable_a3_ti.from_numpy(a3)
    variable_a4_ti.from_numpy(a4)
    variable_a5_ti.from_numpy(a5)
    kget_collision_radius_wq2022_rg_df_k0_a13(
        variable_a1_ti, variable_a2_ti, variable_a3_ti, variable_a4_ti, variable_a5_ti, result_ti
    )
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np
