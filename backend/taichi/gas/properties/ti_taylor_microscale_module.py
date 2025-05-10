"""Taichi implementation of Taylor-microscale utilities."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

@ti.func
def fget_lagrangian_taylor_microscale_time(
    kol: ti.f64, re_lam: ti.f64, acc_var: ti.f64
) -> ti.f64:
    return kol * ti.sqrt((2.0 * re_lam) / (ti.sqrt(15.0) * acc_var))

@ti.func
def fget_taylor_microscale(
    u_rms: ti.f64, nu: ti.f64, eps: ti.f64
) -> ti.f64:
    return u_rms * ti.sqrt((15.0 * nu**2) / eps)

@ti.func
def fget_taylor_microscale_reynolds_number(
    u_rms: ti.f64, lam: ti.f64, nu: ti.f64
) -> ti.f64:
    return (u_rms * lam) / nu

@ti.kernel
def kget_lagrangian_taylor_microscale_time(
    kol: ti.types.ndarray(dtype=ti.f64, ndim=1),
    re_lam: ti.types.ndarray(dtype=ti.f64, ndim=1),
    acc_var: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_lagrangian_taylor_microscale_time(
            kol[i], re_lam[i], acc_var[i]
        )

@ti.kernel
def kget_taylor_microscale(
    u_rms: ti.types.ndarray(dtype=ti.f64, ndim=1),
    nu: ti.types.ndarray(dtype=ti.f64, ndim=1),
    eps: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_taylor_microscale(
            u_rms[i], nu[i], eps[i]
        )

@ti.kernel
def kget_taylor_microscale_reynolds_number(
    u_rms: ti.types.ndarray(dtype=ti.f64, ndim=1),
    lam: ti.types.ndarray(dtype=ti.f64, ndim=1),
    nu: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_taylor_microscale_reynolds_number(
            u_rms[i], lam[i], nu[i]
        )

@register("get_lagrangian_taylor_microscale_time", backend="taichi")
def get_lagrangian_taylor_microscale_time_taichi(kolmogorov_time, re_lambda, accel_variance):
    """Taichi version of Lagrangian Taylor microscale time."""
    if not (isinstance(kolmogorov_time, np.ndarray) and isinstance(re_lambda, np.ndarray) and isinstance(accel_variance, np.ndarray)):
        raise TypeError("Taichi backend expects NumPy arrays for all inputs.")

    kol, re_lam, acc_var = np.atleast_1d(kolmogorov_time), np.atleast_1d(re_lambda), np.atleast_1d(accel_variance)
    n = kol.size

    kol_ti = ti.ndarray(dtype=ti.f64, shape=n)
    re_lam_ti = ti.ndarray(dtype=ti.f64, shape=n)
    acc_var_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    kol_ti.from_numpy(kol)
    re_lam_ti.from_numpy(re_lam)
    acc_var_ti.from_numpy(acc_var)

    kget_lagrangian_taylor_microscale_time(kol_ti, re_lam_ti, acc_var_ti, result_ti)
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np

@register("get_taylor_microscale", backend="taichi")
def get_taylor_microscale_taichi(fluid_rms_velocity, kinematic_viscosity, turbulent_dissipation):
    """Taichi version of Taylor microscale."""
    if not (isinstance(fluid_rms_velocity, np.ndarray) and isinstance(kinematic_viscosity, np.ndarray) and isinstance(turbulent_dissipation, np.ndarray)):
        raise TypeError("Taichi backend expects NumPy arrays for all inputs.")

    u_rms, nu, eps = np.atleast_1d(fluid_rms_velocity), np.atleast_1d(kinematic_viscosity), np.atleast_1d(turbulent_dissipation)
    n = u_rms.size

    u_rms_ti = ti.ndarray(dtype=ti.f64, shape=n)
    nu_ti = ti.ndarray(dtype=ti.f64, shape=n)
    eps_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    u_rms_ti.from_numpy(u_rms)
    nu_ti.from_numpy(nu)
    eps_ti.from_numpy(eps)

    kget_taylor_microscale(u_rms_ti, nu_ti, eps_ti, result_ti)
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np

@register("get_taylor_microscale_reynolds_number", backend="taichi")
def get_taylor_microscale_reynolds_number_taichi(fluid_rms_velocity, taylor_microscale, kinematic_viscosity):
    """Taichi version of Taylor-microscale Reynolds number."""
    if not (isinstance(fluid_rms_velocity, np.ndarray) and isinstance(taylor_microscale, np.ndarray) and isinstance(kinematic_viscosity, np.ndarray)):
        raise TypeError("Taichi backend expects NumPy arrays for all inputs.")

    u_rms, lam, nu = np.atleast_1d(fluid_rms_velocity), np.atleast_1d(taylor_microscale), np.atleast_1d(kinematic_viscosity)
    n = u_rms.size

    u_rms_ti = ti.ndarray(dtype=ti.f64, shape=n)
    lam_ti = ti.ndarray(dtype=ti.f64, shape=n)
    nu_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    u_rms_ti.from_numpy(u_rms)
    lam_ti.from_numpy(lam)
    nu_ti.from_numpy(nu)

    kget_taylor_microscale_reynolds_number(u_rms_ti, lam_ti, nu_ti, result_ti)
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np
