"""Taichi implementation of util.reduced_quantity"""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

# ───────── 3 – scalar Taichi func ─────────
@ti.func
def fget_reduced_value(alpha: ti.f64, beta: ti.f64) -> ti.f64:
    denom = alpha + beta
    return ti.select(denom != 0.0, alpha * beta / denom, 0.0)

# ───────── 4 a – 1-D kernel ─────────
@ti.kernel
def kget_reduced_value(                        # ndim = 1
    alpha: ti.types.ndarray(dtype=ti.f64, ndim=1),
    beta:  ti.types.ndarray(dtype=ti.f64, ndim=1),
    out:   ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(out.shape[0]):
        out[i] = fget_reduced_value(alpha[i], beta[i])

# ───────── 4 b – 2-D self-broadcast kernel ─────────
@ti.kernel
def kget_reduced_self_broadcast(               # ndim = 2
    alpha:  ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=2),
):
    n = alpha.shape[0]
    for i, j in ti.ndrange(n, n):
        result[i, j] = fget_reduced_value(alpha[i], alpha[j])

# ───────── 5 a – wrapper for element-wise version ─────────
@register("get_reduced_value", backend="taichi")
def ti_get_reduced_value(alpha, beta):
    if not (isinstance(alpha, np.ndarray) and isinstance(beta, np.ndarray)):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")
    a1 = np.asarray(alpha, dtype=np.float64).ravel()
    a2 = np.asarray(beta,  dtype=np.float64).ravel()
    if a1.size != a2.size:
        raise ValueError("Alpha and beta must have identical shapes.")
    n = a1.size
    alpha_ti = ti.ndarray(dtype=ti.f64, shape=n)
    beta_ti  = ti.ndarray(dtype=ti.f64, shape=n)
    out_ti   = ti.ndarray(dtype=ti.f64, shape=n)
    alpha_ti.from_numpy(a1)
    beta_ti.from_numpy(a2)
    kget_reduced_value(alpha_ti, beta_ti, out_ti)
    res = out_ti.to_numpy().reshape(np.asarray(alpha).shape)
    return res.item() if res.size == 1 else res

# ───────── 5 b – wrapper for self-broadcast version ─────────
@register("get_reduced_self_broadcast", backend="taichi")
def ti_get_reduced_self_broadcast(alpha):
    if not isinstance(alpha, np.ndarray):
        raise TypeError("Taichi backend expects a NumPy array.")
    a = np.asarray(alpha, dtype=np.float64).ravel()
    n = a.size
    alpha_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti   = ti.ndarray(dtype=ti.f64, shape=(n, n))
    alpha_ti.from_numpy(a)
    kget_reduced_self_broadcast(alpha_ti, res_ti)
    return res_ti.to_numpy()
