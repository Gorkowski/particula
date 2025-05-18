"""
Taichi-backed implementation of ``util.reduced_quantity``.

This module offers scalar, 1-D vectorised and 2-D self-broadcast Taichi
kernels as well as NumPy-friendly wrapper functions.  All routines return
the reduced (harmonic-mean) quantity

    r = (α × β) / (α + β)   if α + β ≠ 0  
    r = 0                   otherwise.

Examples:
    ```py title="Quick usage"
    import numpy as np
    from particula.backend import use_backend

    use_backend("taichi")
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    print(particula.get_reduced_value(a, b))
    ```
"""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

# ───────── 3 – scalar Taichi func ─────────
@ti.func
def fget_reduced_value(alpha: ti.f64, beta: ti.f64) -> ti.f64:
    """
    Return the harmonic mean of two scalars.

    r = (α × β) / (α + β)  if α + β ≠ 0  
    r = 0                  otherwise.

    Arguments:
        - alpha : First scalar α.
        - beta : Second scalar β.

    Returns:
        - Reduced value r.
    """
    denominator = alpha + beta
    return ti.select(denominator != 0.0,
                     alpha * beta / denominator,
                     0.0)

# ───────── 4 a – 1-D kernel ─────────
@ti.kernel
def kget_reduced_value(                        # ndim = 1
    alpha: ti.types.ndarray(dtype=ti.f64, ndim=1),
    beta: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Element-wise wrapper over ``fget_reduced_value``.

    Arguments:
        - alpha : 1-D array of α values.
        - beta : 1-D array of β values (same shape as α).
        - result_array : Output array (pre-allocated, same shape).

    Returns:
        - None (writes in-place).
    """
    for i in range(result_array.shape[0]):
        result_array[i] = fget_reduced_value(alpha[i], beta[i])

# ───────── 4 b – 2-D self-broadcast kernel ─────────
@ti.kernel
def kget_reduced_self_broadcast(               # ndim = 2
    alpha: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result_array: ti.types.ndarray(dtype=ti.f64, ndim=2),
):
    """
    Fill a symmetric matrix with pairwise reduced values.

    Arguments:
        - alpha : 1-D array of scalars.
        - result_array : 2-D (n × n) output matrix.

    Returns:
        - None.
    """
    n_elements = alpha.shape[0]
    for i, j in ti.ndrange(n_elements, n_elements):
        result_array[i, j] = fget_reduced_value(alpha[i], alpha[j])

# ───────── 5 a – wrapper for element-wise version ─────────
@register("get_reduced_value", backend="taichi")
def ti_get_reduced_value(alpha, beta):
    """
    NumPy interface to ``kget_reduced_value``.

    Arguments:
        - alpha : NumPy scalar/array of α.
        - beta : NumPy scalar/array of β (same shape).

    Returns:
        - Scalar or ndarray with reduced values.
    """
    if not (isinstance(alpha, np.ndarray) and isinstance(beta, np.ndarray)):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")
    alpha_flat = np.asarray(alpha, dtype=np.float64).ravel()
    beta_flat = np.asarray(beta, dtype=np.float64).ravel()
    if alpha_flat.size != beta_flat.size:
        raise ValueError("Alpha and beta must have identical shapes.")
    n_elements = alpha_flat.size
    alpha_ti_field = ti.ndarray(dtype=ti.f64, shape=n_elements)
    beta_ti_field = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_ti_field = ti.ndarray(dtype=ti.f64, shape=n_elements)
    alpha_ti_field.from_numpy(alpha_flat)
    beta_ti_field.from_numpy(beta_flat)
    kget_reduced_value(alpha_ti_field, beta_ti_field, result_ti_field)
    result = result_ti_field.to_numpy().reshape(np.asarray(alpha).shape)
    return result.item() if result.size == 1 else result

# ───────── 5 b – wrapper for self-broadcast version ─────────
@register("get_reduced_self_broadcast", backend="taichi")
def ti_get_reduced_self_broadcast(alpha):
    """
    NumPy interface to ``kget_reduced_self_broadcast``.

    Arguments:
        - alpha : NumPy 1-D array of α.

    Returns:
        - 2-D ndarray of pairwise reduced values.
    """
    if not isinstance(alpha, np.ndarray):
        raise TypeError("Taichi backend expects a NumPy array.")
    alpha_flat = np.asarray(alpha, dtype=np.float64).ravel()
    n_elements = alpha_flat.size
    alpha_ti_field = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_ti_field = ti.ndarray(
        dtype=ti.f64,
        shape=(n_elements, n_elements),
    )
    alpha_ti_field.from_numpy(alpha_flat)
    kget_reduced_self_broadcast(alpha_ti_field, result_ti_field)
    return result_ti_field.to_numpy()
