"""Taichi-accelerated κ-Köhler volume conversions."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

# ── 3 ▸ element-wise Taichi functions ──────────────────────────
@ti.func
def fget_solute_volume_from_kappa(v_tot: ti.f64, kappa: ti.f64, aw: ti.f64) -> ti.f64:
    kappa = ti.max(kappa, 1e-16)
    if aw <= 1e-16:
        return v_tot
    vol_factor = (aw - 1.0) / (aw * (1.0 - kappa - 1.0 / aw))
    return v_tot * vol_factor


@ti.func
def fget_water_volume_from_kappa(v_sol: ti.f64, kappa: ti.f64, aw: ti.f64) -> ti.f64:
    aw = ti.min(aw, 1.0 - 1e-16)
    if aw <= 1e-16:
        return 0.0
    return v_sol * kappa / (1.0 / aw - 1.0)


@ti.func
def fget_kappa_from_volumes(v_sol: ti.f64, v_wat: ti.f64, aw: ti.f64) -> ti.f64:
    aw = ti.min(aw, 1.0 - 1e-16)
    return (1.0 / aw - 1.0) * v_wat / v_sol


@ti.func
def fget_water_volume_in_mixture(v_sol_dry: ti.f64, phi_w: ti.f64) -> ti.f64:
    return phi_w * v_sol_dry / (1.0 - phi_w)


# ── 4 ▸ vectorised kernels ─────────────────────────────────────
@ti.kernel
def kget_solute_volume_from_kappa(
    vt: ti.types.ndarray(dtype=ti.f64, ndim=1),
    kp: ti.types.ndarray(dtype=ti.f64, ndim=1),
    aw: ti.types.ndarray(dtype=ti.f64, ndim=1),
    res: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(res.shape[0]):
        res[i] = fget_solute_volume_from_kappa(vt[i], kp[i], aw[i])


@ti.kernel
def kget_water_volume_from_kappa(
    vs: ti.types.ndarray(dtype=ti.f64, ndim=1),
    kp: ti.types.ndarray(dtype=ti.f64, ndim=1),
    aw: ti.types.ndarray(dtype=ti.f64, ndim=1),
    res: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(res.shape[0]):
        res[i] = fget_water_volume_from_kappa(vs[i], kp[i], aw[i])


@ti.kernel
def kget_kappa_from_volumes(
    vs: ti.types.ndarray(dtype=ti.f64, ndim=1),
    vw: ti.types.ndarray(dtype=ti.f64, ndim=1),
    aw: ti.types.ndarray(dtype=ti.f64, ndim=1),
    res: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(res.shape[0]):
        res[i] = fget_kappa_from_volumes(vs[i], vw[i], aw[i])


@ti.kernel
def kget_water_volume_in_mixture(
    vsd: ti.types.ndarray(dtype=ti.f64, ndim=1),
    phi: ti.types.ndarray(dtype=ti.f64, ndim=1),
    res: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(res.shape[0]):
        res[i] = fget_water_volume_in_mixture(vsd[i], phi[i])


# ── 5 ▸ public wrappers with backend registration ──────────────
def _wrap(kernel, *arrays):
    a_np = [np.atleast_1d(x).astype(np.float64) for x in arrays]
    n = a_np[0].size
    ti_in = [ti.ndarray(dtype=ti.f64, shape=n) for _ in arrays]
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    for buf, arr in zip(ti_in, a_np):
        buf.from_numpy(arr)
    kernel(*ti_in, res_ti)
    res_np = res_ti.to_numpy()
    return res_np.item() if res_np.size == 1 else res_np


@register("get_solute_volume_from_kappa", backend="taichi")
def ti_get_solute_volume_from_kappa(volume_total, kappa, water_activity):
    if not (isinstance(volume_total, np.ndarray) or np.isscalar(volume_total)):
        raise TypeError("Taichi backend expects NumPy array or scalar.")
    return _wrap(kget_solute_volume_from_kappa, volume_total, kappa, water_activity)


@register("get_water_volume_from_kappa", backend="taichi")
def ti_get_water_volume_from_kappa(volume_solute, kappa, water_activity):
    if not (isinstance(volume_solute, np.ndarray) or np.isscalar(volume_solute)):
        raise TypeError("Taichi backend expects NumPy array or scalar.")
    return _wrap(kget_water_volume_from_kappa, volume_solute, kappa, water_activity)


@register("get_kappa_from_volumes", backend="taichi")
def ti_get_kappa_from_volumes(volume_solute, volume_water, water_activity):
    if not (isinstance(volume_solute, np.ndarray) or np.isscalar(volume_solute)):
        raise TypeError("Taichi backend expects NumPy array or scalar.")
    return _wrap(kget_kappa_from_volumes, volume_solute, volume_water, water_activity)


@register("get_water_volume_in_mixture", backend="taichi")
def ti_get_water_volume_in_mixture(volume_solute_dry, volume_fraction_water):
    if not (isinstance(volume_solute_dry, np.ndarray) or np.isscalar(volume_solute_dry)):
        raise TypeError("Taichi backend expects NumPy array or scalar.")
    return _wrap(kget_water_volume_in_mixture, volume_solute_dry, volume_fraction_water)
