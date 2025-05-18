"""Taichi implementation of util.reduced_quantity"""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

# ───────── 3 – scalar Taichi func ─────────
@ti.func
def fget_reduced_value(alpha: ti.f64, beta: ti.f64) -> ti.f64:
    denominator = alpha + beta
    return ti.select(denominator != 0.0,
                     alpha * beta / denominator,
                     0.0)

# ───────── 4 a – 1-D kernel ─────────
@ti.kernel
def kget_reduced_value(                        # ndim = 1
    alpha: ti.types.ndarray(dtype=ti.f64, ndim=1),
    beta:  ti.types.ndarray(dtype=ti.f64, ndim=1),
    result_array:   ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result_array.shape[0]):
        result_array[i] = fget_reduced_value(alpha[i], beta[i])

# ───────── 4 b – 2-D self-broadcast kernel ─────────
@ti.kernel
def kget_reduced_self_broadcast(               # ndim = 2
    alpha:  ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=2),
):
    n_elements = alpha.shape[0]
    for i, j in ti.ndrange(n_elements, n_elements):
        result[i, j] = fget_reduced_value(alpha[i], alpha[j])

# ───────── 5 a – wrapper for element-wise version ─────────
@register("get_reduced_value", backend="taichi")
def ti_get_reduced_value(alpha, beta):
    if not (isinstance(alpha, np.ndarray) and isinstance(beta, np.ndarray)):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")
    alpha_flat = np.asarray(alpha, dtype=np.float64).ravel()
    beta_flat  = np.asarray(beta,  dtype=np.float64).ravel()
    if alpha_flat.size != beta_flat.size:
        raise ValueError("Alpha and beta must have identical shapes.")
    n_elements = alpha_flat.size
    alpha_ti_field = ti.ndarray(dtype=ti.f64, shape=n_elements)
    beta_ti_field  = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_ti_field = ti.ndarray(dtype=ti.f64, shape=n_elements)
    alpha_ti_field.from_numpy(alpha_flat)
    beta_ti_field.from_numpy(beta_flat)
    kget_reduced_value(alpha_ti_field, beta_ti_field, result_ti_field)
    result = result_ti_field.to_numpy().reshape(np.asarray(alpha).shape)
    return result.item() if result.size == 1 else result

# ───────── 5 b – wrapper for self-broadcast version ─────────
@register("get_reduced_self_broadcast", backend="taichi")
def ti_get_reduced_self_broadcast(alpha):
    if not isinstance(alpha, np.ndarray):
        raise TypeError("Taichi backend expects a NumPy array.")
    alpha_flat = np.asarray(alpha, dtype=np.float64).ravel()
    n_elements = alpha_flat.size
    alpha_ti_field = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_ti_field = ti.ndarray(dtype=ti.f64,
                                 shape=(n_elements, n_elements))
    alpha_ti_field.from_numpy(alpha_flat)
    kget_reduced_self_broadcast(alpha_ti_field, result_ti_field)
    return result_ti_field.to_numpy()
