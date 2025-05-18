"""Taichi-accelerated κ-Köhler volume conversions."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

# ── 3 ▸ element-wise Taichi functions ──────────────────────────
@ti.func
def fget_solute_volume_from_kappa(
    volume_total: ti.f64, kappa: ti.f64, water_activity: ti.f64
) -> ti.f64:
    """
    Compute solute (dry) volume Vₛ using the κ-Köhler relation.

    Vₛ = Vₜ × (a_w − 1) ∕ [a_w × (1 − κ − 1 ∕ a_w)]

    Arguments:
        - volume_total : Total particle volume Vₜ [m³].
        - kappa : Hygroscopicity κ (dimensionless, ≥ 0).
        - water_activity : Water activity a_w (0 < a_w ≤ 1).

    Returns:
        - volume_solute : Solute volume Vₛ [m³].

    Examples:
        ```py title="Scalar example"
        fget_solute_volume_from_kappa(1e-18, 0.3, 0.9)
        ```

    References:
        - Petters & Kreidenweis, Atmos. Chem. Phys., 2007.
    """
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
    """
    Compute water volume V_w from solute volume and κ-Köhler relation.

    V_w = Vₛ × κ ∕ (1 ∕ a_w − 1)

    Arguments:
        - volume_solute : Solute (dry) volume Vₛ [m³].
        - kappa : Hygroscopicity κ (dimensionless, ≥ 0).
        - water_activity : Water activity a_w (0 < a_w ≤ 1).

    Returns:
        - volume_water : Water volume V_w [m³].

    Examples:
        ```py title="Scalar example"
        fget_water_volume_from_kappa(1e-18, 0.3, 0.9)
        ```

    References:
        - Petters & Kreidenweis, Atmos. Chem. Phys., 2007.
    """
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
    """
    Compute κ from solute and water volumes and water activity.

    κ = (1 ∕ a_w − 1) × V_w ∕ Vₛ

    Arguments:
        - volume_solute : Solute (dry) volume Vₛ [m³].
        - volume_water : Water volume V_w [m³].
        - water_activity : Water activity a_w (0 < a_w ≤ 1).

    Returns:
        - kappa : Hygroscopicity κ (dimensionless).

    Examples:
        ```py title="Scalar example"
        fget_kappa_from_volumes(1e-18, 2e-18, 0.9)
        ```

    References:
        - Petters & Kreidenweis, Atmos. Chem. Phys., 2007.
    """
    water_activity = ti.min(water_activity, 1.0 - 1e-16)
    return (1.0 / water_activity - 1.0) * volume_water / volume_solute


@ti.func
def fget_water_volume_in_mixture(
    volume_solute_dry: ti.f64, volume_fraction_water: ti.f64
) -> ti.f64:
    """
    Compute water volume in a mixture from solute dry volume and water fraction.

    V_w = f_w × Vₛ,₍dry₎ ∕ (1 − f_w)

    Arguments:
        - volume_solute_dry : Dry solute volume Vₛ,₍dry₎ [m³].
        - volume_fraction_water : Water volume fraction f_w (0 ≤ f_w < 1).

    Returns:
        - volume_water : Water volume V_w [m³].

    Examples:
        ```py title="Scalar example"
        fget_water_volume_in_mixture(1e-18, 0.5)
        ```

    References:
        - Petters & Kreidenweis, Atmos. Chem. Phys., 2007.
    """
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
    """
    Vectorized Taichi kernel for solute volume from κ-Köhler relation.

    Computes Vₛ = Vₜ × (a_w − 1) ∕ [a_w × (1 − κ − 1 ∕ a_w)] for arrays.

    Arguments:
        - volume_total : Array of total particle volumes Vₜ [m³].
        - kappa : Array of hygroscopicity κ (dimensionless).
        - water_activity : Water activity a_w (scalar, 0 < a_w ≤ 1).
        - result : Output array for solute volumes Vₛ [m³].

    Returns:
        - None (results written to result array).

    Examples:
        ```py title="Array example"
        kget_solute_volume_from_kappa(vols, kappas, 0.9, result)
        ```

    References:
        - Petters & Kreidenweis, Atmos. Chem. Phys., 2007.
    """
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
    """
    Vectorized Taichi kernel for water volume from κ-Köhler relation.

    Computes V_w = Vₛ × κ ∕ (1 ∕ a_w − 1) for arrays.

    Arguments:
        - volume_solute : Array of solute (dry) volumes Vₛ [m³].
        - kappa : Array of hygroscopicity κ (dimensionless).
        - water_activity : Water activity a_w (scalar, 0 < a_w ≤ 1).
        - result : Output array for water volumes V_w [m³].

    Returns:
        - None (results written to result array).

    Examples:
        ```py title="Array example"
        kget_water_volume_from_kappa(solutes, kappas, 0.9, result)
        ```

    References:
        - Petters & Kreidenweis, Atmos. Chem. Phys., 2007.
    """
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
    """
    Vectorized Taichi kernel for κ from solute and water volumes.

    Computes κ = (1 ∕ a_w − 1) × V_w ∕ Vₛ for arrays.

    Arguments:
        - volume_solute : Array of solute (dry) volumes Vₛ [m³].
        - volume_water : Array of water volumes V_w [m³].
        - water_activity : Water activity a_w (scalar, 0 < a_w ≤ 1).
        - result : Output array for κ (dimensionless).

    Returns:
        - None (results written to result array).

    Examples:
        ```py title="Array example"
        kget_kappa_from_volumes(solutes, waters, 0.9, result)
        ```

    References:
        - Petters & Kreidenweis, Atmos. Chem. Phys., 2007.
    """
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
    """
    Vectorized Taichi kernel for water volume in a mixture.

    Computes V_w = f_w × Vₛ,₍dry₎ ∕ (1 − f_w) for arrays.

    Arguments:
        - volume_solute_dry : Array of dry solute volumes Vₛ,₍dry₎ [m³].
        - volume_fraction_water : Array of water volume fractions f_w.
        - result : Output array for water volumes V_w [m³].

    Returns:
        - None (results written to result array).

    Examples:
        ```py title="Array example"
        kget_water_volume_in_mixture(solutes, fractions, result)
        ```

    References:
        - Petters & Kreidenweis, Atmos. Chem. Phys., 2007.
    """
    for i in range(result.shape[0]):
        result[i] = fget_water_volume_in_mixture(
            volume_solute_dry[i], volume_fraction_water[i]
        )


# ── 5 ▸ public wrappers with backend registration ──────────────
def _wrap(kernel, *arrays, scalar_args=()):
    """
    Broadcast input arrays, launch Taichi kernel, and reshape result.

    This function prepares NumPy arrays for Taichi kernels, broadcasts
    shapes, flattens, and allocates Taichi ndarrays. It then calls the
    kernel and reshapes the result to match the input.

    Arguments:
        - kernel : Taichi kernel function to call.
        - *arrays : Input arrays (NumPy or scalars) to broadcast.
        - scalar_args : Tuple of scalar arguments to pass to kernel.

    Returns:
        - result : NumPy array or scalar with the kernel output.

    Examples:
        ```py title="Example"
        _wrap(kget_solute_volume_from_kappa, vols, kappas, scalar_args=(0.9,))
        ```

    References:
        - Taichi documentation: https://docs.taichi-lang.org/
    """
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
    """
    Public wrapper for Taichi backend: solute volume from κ-Köhler relation.

    Arguments:
        - volume_total : Total particle volume(s) Vₜ [m³] (scalar or array).
        - kappa : Hygroscopicity κ (scalar or array).
        - water_activity : Water activity a_w (scalar, 0 < a_w ≤ 1).

    Returns:
        - volume_solute : Solute volume(s) Vₛ [m³] (scalar or array).

    Examples:
        ```py title="Example"
        ti_get_solute_volume_from_kappa(1e-18, 0.3, 0.9)
        ```

    References:
        - Petters & Kreidenweis, Atmos. Chem. Phys., 2007.
    """
    if isinstance(water_activity, np.ndarray) and water_activity.size != 1:
        raise TypeError("water_activity must be a scalar for Taichi backend.")
    if not all(
        isinstance(x, (np.ndarray, float, int))
        for x in (volume_total, kappa)
    ):
        raise TypeError("volume_total and kappa must be scalars or NumPy arrays.")
    return _wrap(
        kget_solute_volume_from_kappa,
        volume_total,
        kappa,
        scalar_args=(water_activity,),
    )


@register("get_water_volume_from_kappa", backend="taichi")
def ti_get_water_volume_from_kappa(volume_solute, kappa, water_activity):
    """
    Public wrapper for Taichi backend: water volume from κ-Köhler relation.

    Arguments:
        - volume_solute : Solute (dry) volume(s) Vₛ [m³] (scalar or array).
        - kappa : Hygroscopicity κ (scalar or array).
        - water_activity : Water activity a_w (scalar, 0 < a_w ≤ 1).

    Returns:
        - volume_water : Water volume(s) V_w [m³] (scalar or array).

    Examples:
        ```py title="Example"
        ti_get_water_volume_from_kappa(1e-18, 0.3, 0.9)
        ```

    References:
        - Petters & Kreidenweis, Atmos. Chem. Phys., 2007.
    """
    if isinstance(water_activity, np.ndarray) and water_activity.size != 1:
        raise TypeError("water_activity must be a scalar for Taichi backend.")
    if not all(
        isinstance(x, (np.ndarray, float, int))
        for x in (volume_solute, kappa)
    ):
        raise TypeError("volume_solute and kappa must be scalars or NumPy arrays.")
    return _wrap(
        kget_water_volume_from_kappa,
        volume_solute,
        kappa,
        scalar_args=(water_activity,),
    )


@register("get_kappa_from_volumes", backend="taichi")
def ti_get_kappa_from_volumes(volume_solute, volume_water, water_activity):
    """
    Public wrapper for Taichi backend: κ from solute and water volumes.

    Arguments:
        - volume_solute : Solute (dry) volume(s) Vₛ [m³] (scalar or array).
        - volume_water : Water volume(s) V_w [m³] (scalar or array).
        - water_activity : Water activity a_w (scalar, 0 < a_w ≤ 1).

    Returns:
        - kappa : Hygroscopicity κ (scalar or array).

    Examples:
        ```py title="Example"
        ti_get_kappa_from_volumes(1e-18, 2e-18, 0.9)
        ```

    References:
        - Petters & Kreidenweis, Atmos. Chem. Phys., 2007.
    """
    if isinstance(water_activity, np.ndarray) and water_activity.size != 1:
        raise TypeError("water_activity must be a scalar for Taichi backend.")
    if not all(
        isinstance(x, (np.ndarray, float, int))
        for x in (volume_solute, volume_water)
    ):
        raise TypeError("volume_solute and volume_water must be scalars or NumPy arrays.")
    return _wrap(
        kget_kappa_from_volumes,
        volume_solute,
        volume_water,
        scalar_args=(water_activity,),
    )


@register("get_water_volume_in_mixture", backend="taichi")
def ti_get_water_volume_in_mixture(volume_solute_dry, volume_fraction_water):
    """
    Public wrapper for Taichi backend: water volume in a mixture.

    Arguments:
        - volume_solute_dry : Dry solute volume(s) Vₛ,₍dry₎ [m³] (scalar or array).
        - volume_fraction_water : Water volume fraction(s) f_w (scalar or array).

    Returns:
        - volume_water : Water volume(s) V_w [m³] (scalar or array).

    Examples:
        ```py title="Example"
        ti_get_water_volume_in_mixture(1e-18, 0.5)
        ```

    References:
        - Petters & Kreidenweis, Atmos. Chem. Phys., 2007.
    """
    if not all(
        isinstance(x, (np.ndarray, float, int))
        for x in (volume_solute_dry, volume_fraction_water)
    ):
        raise TypeError("Taichi backend expects NumPy arrays or scalars.")
    return _wrap(kget_water_volume_in_mixture, volume_solute_dry, volume_fraction_water)
