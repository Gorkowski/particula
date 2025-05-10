"""Taichi implementation of integral–scale utilities."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

# ── 3. element-wise Taichi funcs ────────────────────────────────────────────
@ti.func
def fget_lagrangian_integral_time(u_rms: ti.f64, eps: ti.f64) -> ti.f64:      # noqa: N802
    return (u_rms * u_rms) / eps

@ti.func
def fget_eulerian_integral_length(u_rms: ti.f64, eps: ti.f64) -> ti.f64:      # noqa: N802
    return 0.5 * u_rms * u_rms * u_rms / eps

# ── 4. vectorised kernels ───────────────────────────────────────────────────
@ti.kernel
def kget_lagrangian_integral_time(                                           # noqa: N802
    u_rms: ti.types.ndarray(dtype=ti.f64, ndim=1),
    eps: ti.types.ndarray(dtype=ti.f64, ndim=1),
    res: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(res.shape[0]):
        res[i] = fget_lagrangian_integral_time(u_rms[i], eps[i])

@ti.kernel
def kget_eulerian_integral_length(                                           # noqa: N802
    u_rms: ti.types.ndarray(dtype=ti.f64, ndim=1),
    eps: ti.types.ndarray(dtype=ti.f64, ndim=1),
    res: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(res.shape[0]):
        res[i] = fget_eulerian_integral_length(u_rms[i], eps[i])

# ── 5. public wrappers with backend registration ────────────────────────────
@register("get_lagrangian_integral_time", backend="taichi")
def get_lagrangian_integral_time_taichi(fluid_rms_velocity, turbulent_dissipation):  # noqa: D401
    if not (isinstance(fluid_rms_velocity, np.ndarray) and isinstance(turbulent_dissipation, np.ndarray)):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")

    u, eps = np.atleast_1d(fluid_rms_velocity), np.atleast_1d(turbulent_dissipation)
    n = u.size

    u_ti, eps_ti = [ti.ndarray(dtype=ti.f64, shape=n) for _ in range(2)]
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    u_ti.from_numpy(u)
    eps_ti.from_numpy(eps)

    kget_lagrangian_integral_time(u_ti, eps_ti, res_ti)

    out = res_ti.to_numpy()
    return out.item() if out.size == 1 else out

@register("get_eulerian_integral_length", backend="taichi")
def get_eulerian_integral_length_taichi(fluid_rms_velocity, turbulent_dissipation):  # noqa: D401
    if not (isinstance(fluid_rms_velocity, np.ndarray) and isinstance(turbulent_dissipation, np.ndarray)):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")

    u, eps = np.atleast_1d(fluid_rms_velocity), np.atleast_1d(turbulent_dissipation)
    n = u.size

    u_ti, eps_ti = [ti.ndarray(dtype=ti.f64, shape=n) for _ in range(2)]
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    u_ti.from_numpy(u)
    eps_ti.from_numpy(eps)

    kget_eulerian_integral_length(u_ti, eps_ti, res_ti)

    out = res_ti.to_numpy()
    return out.item() if out.size == 1 else out
