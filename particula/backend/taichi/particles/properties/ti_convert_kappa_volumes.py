"""Taichi-accelerated κ-Köhler volume conversions."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

# ── 3 ▸ element-wise Taichi functions ──────────────────────────
@ti.func
def fget_solute_volume_from_kappa(
    volume_total: ti.f64, kappa: ti.f64, water_activity: ti.f64
) -> ti.f64:
    kappa = ti.max(kappa, 1e-16)
    # default: water_activity very small ⇒ whole volume is solute
    vol_factor = 1.0
    if water_activity > 1e-16:
        vol_factor = (water_activity - 1.0) / (
            water_activity * (1.0 - kappa - 1.0 / water_activity)
        )
    return volume_total * vol_factor


@ti.func
def fget_water_volume_from_kappa(
    volume_solute: ti.f64, kappa: ti.f64, water_activity: ti.f64
) -> ti.f64:
    water_activity = ti.min(water_activity, 1.0 - 1e-16)
    result = 0.0
    if water_activity > 1e-16:
        result = (
            volume_solute * kappa / (1.0 / water_activity - 1.0)
        )
    return result


@ti.func
def fget_kappa_from_volumes(
    volume_solute: ti.f64, volume_water: ti.f64, water_activity: ti.f64
) -> ti.f64:
    water_activity = ti.min(water_activity, 1.0 - 1e-16)
    return (1.0 / water_activity - 1.0) * volume_water / volume_solute


@ti.func
def fget_water_volume_in_mixture(
    volume_solute_dry: ti.f64, volume_fraction_water: ti.f64
) -> ti.f64:
    return (
        volume_fraction_water * volume_solute_dry / (1.0 - volume_fraction_water)
    )


# ── 4 ▸ vectorised kernels ─────────────────────────────────────
@ti.kernel
def kget_solute_volume_from_kappa(
    volume_total: ti.types.ndarray(dtype=ti.f64, ndim=1),
    kappa: ti.types.ndarray(dtype=ti.f64, ndim=1),
    water_activity: ti.f64,
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_solute_volume_from_kappa(
            volume_total[i], kappa[i], water_activity
        )


@ti.kernel
def kget_water_volume_from_kappa(
    volume_solute: ti.types.ndarray(dtype=ti.f64, ndim=1),
    kappa: ti.types.ndarray(dtype=ti.f64, ndim=1),
    water_activity: ti.f64,
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_water_volume_from_kappa(
            volume_solute[i], kappa[i], water_activity
        )


@ti.kernel
def kget_kappa_from_volumes(
    volume_solute: ti.types.ndarray(dtype=ti.f64, ndim=1),
    volume_water: ti.types.ndarray(dtype=ti.f64, ndim=1),
    water_activity: ti.f64,
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_kappa_from_volumes(
            volume_solute[i], volume_water[i], water_activity
        )


@ti.kernel
def kget_water_volume_in_mixture(
    volume_solute_dry: ti.types.ndarray(dtype=ti.f64, ndim=1),
    volume_fraction_water: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_water_volume_in_mixture(
            volume_solute_dry[i], volume_fraction_water[i]
        )


# ── 5 ▸ public wrappers with backend registration ──────────────
def _wrap(kernel, *arrays, scalar_args=()):
    """Broadcast inputs, launch kernel, reshape result."""
    b_arrays = np.broadcast_arrays(
        *[np.asarray(a, dtype=np.float64) for a in arrays]
    )
    flat = [a.ravel() for a in b_arrays]
    n = flat[0].size

    ti_in = [ti.ndarray(dtype=ti.f64, shape=n) for _ in flat]
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    for buf, arr in zip(ti_in, flat):
        buf.from_numpy(arr)

    scalars = [np.float64(s) for s in scalar_args]  # ensure f64 scalars
    kernel(*ti_in, *scalars, result_ti)

    result_np = result_ti.to_numpy().reshape(b_arrays[0].shape)
    return result_np.item() if result_np.size == 1 else result_np


@register("get_solute_volume_from_kappa", backend="taichi")
def ti_get_solute_volume_from_kappa(volume_total, kappa, water_activity):
    if isinstance(water_activity, np.ndarray) and water_activity.size != 1:
        raise TypeError("water_activity must be a scalar for Taichi backend.")
    if not all(isinstance(x, (np.ndarray, float, int)) for x in (volume_total, kappa)):
        raise TypeError("volume_total and kappa must be scalars or NumPy arrays.")
    return _wrap(
        kget_solute_volume_from_kappa,
        volume_total,
        kappa,
        scalar_args=(water_activity,),
    )


@register("get_water_volume_from_kappa", backend="taichi")
def ti_get_water_volume_from_kappa(volume_solute, kappa, water_activity):
    if isinstance(water_activity, np.ndarray) and water_activity.size != 1:
        raise TypeError("water_activity must be a scalar for Taichi backend.")
    if not all(isinstance(x, (np.ndarray, float, int)) for x in (volume_solute, kappa)):
        raise TypeError("volume_solute and kappa must be scalars or NumPy arrays.")
    return _wrap(
        kget_water_volume_from_kappa,
        volume_solute,
        kappa,
        scalar_args=(water_activity,),
    )


@register("get_kappa_from_volumes", backend="taichi")
def ti_get_kappa_from_volumes(volume_solute, volume_water, water_activity):
    if isinstance(water_activity, np.ndarray) and water_activity.size != 1:
        raise TypeError("water_activity must be a scalar for Taichi backend.")
    if not all(isinstance(x, (np.ndarray, float, int)) for x in (volume_solute, volume_water)):
        raise TypeError("volume_solute and volume_water must be scalars or NumPy arrays.")
    return _wrap(
        kget_kappa_from_volumes,
        volume_solute,
        volume_water,
        scalar_args=(water_activity,),
    )


@register("get_water_volume_in_mixture", backend="taichi")
def ti_get_water_volume_in_mixture(volume_solute_dry, volume_fraction_water):
    if not all(isinstance(x, (np.ndarray, float, int)) for x in (volume_solute_dry, volume_fraction_water)):
        raise TypeError("Taichi backend expects NumPy arrays or scalars.")
    return _wrap(kget_water_volume_in_mixture, volume_solute_dry, volume_fraction_water)
