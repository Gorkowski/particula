"""Taichi-accelerated κ-Köhler volume conversions."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

# ── 3 ▸ element-wise Taichi functions ──────────────────────────
@ti.func
def fget_solute_volume_from_kappa(v_tot: ti.f64, kappa: ti.f64, aw: ti.f64) -> ti.f64:
    kappa = ti.max(kappa, 1e-16)
    # default: aw very small ⇒ whole volume is solute
    vol_factor = 1.0
    if aw > 1e-16:
        vol_factor = (aw - 1.0) / (aw * (1.0 - kappa - 1.0 / aw))
    return v_tot * vol_factor


@ti.func
def fget_water_volume_from_kappa(v_sol: ti.f64, kappa: ti.f64, aw: ti.f64) -> ti.f64:
    aw = ti.min(aw, 1.0 - 1e-16)
    res = 0.0
    if aw > 1e-16:
        res = v_sol * kappa / (1.0 / aw - 1.0)
    return res


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
    """Broadcast inputs, launch kernel, reshape result."""
    # 5 a – convert to float64 nd-arrays and broadcast to common shape
    b_arrays = np.broadcast_arrays(
        *[np.asarray(a, dtype=np.float64) for a in arrays]
    )
    flat = [a.ravel() for a in b_arrays]          # 1-D buffers for Taichi
    n = flat[0].size

    # 5 b – allocate Taichi NDArrays
    ti_in = [ti.ndarray(dtype=ti.f64, shape=n) for _ in flat]
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    for buf, arr in zip(ti_in, flat):
        buf.from_numpy(arr)

    # 5 c – launch the kernel
    kernel(*ti_in, result_ti)

    # 5 d – reshape back to broadcasted shape and unwrap scalar
    result_np = result_ti.to_numpy().reshape(b_arrays[0].shape)
    return result_np.item() if result_np.size == 1 else result_np


@register("get_solute_volume_from_kappa", backend="taichi")
def ti_get_solute_volume_from_kappa(volume_total, kappa, water_activity):
    if not all(isinstance(x, (np.ndarray, float, int)) for x in (volume_total, kappa, water_activity)):
        raise TypeError("Taichi backend expects NumPy arrays or scalars.")
    return _wrap(kget_solute_volume_from_kappa, volume_total, kappa, water_activity)


@register("get_water_volume_from_kappa", backend="taichi")
def ti_get_water_volume_from_kappa(volume_solute, kappa, water_activity):
    if not all(isinstance(x, (np.ndarray, float, int)) for x in (volume_solute, kappa, water_activity)):
        raise TypeError("Taichi backend expects NumPy arrays or scalars.")
    return _wrap(kget_water_volume_from_kappa, volume_solute, kappa, water_activity)


@register("get_kappa_from_volumes", backend="taichi")
def ti_get_kappa_from_volumes(volume_solute, volume_water, water_activity):
    if not all(isinstance(x, (np.ndarray, float, int)) for x in (volume_solute, volume_water, water_activity)):
        raise TypeError("Taichi backend expects NumPy arrays or scalars.")
    return _wrap(kget_kappa_from_volumes, volume_solute, volume_water, water_activity)


@register("get_water_volume_in_mixture", backend="taichi")
def ti_get_water_volume_in_mixture(volume_solute_dry, volume_fraction_water):
    if not all(isinstance(x, (np.ndarray, float, int)) for x in (volume_solute_dry, volume_fraction_water)):
        raise TypeError("Taichi backend expects NumPy arrays or scalars.")
    return _wrap(kget_water_volume_in_mixture, volume_solute_dry, volume_fraction_water)
