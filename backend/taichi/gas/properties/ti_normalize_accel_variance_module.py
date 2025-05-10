"""Taichi version – Ao 2008 normalised acceleration variance."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

@ti.func
def fget_normalized_accel_variance_ao2008(
    re_lambda: ti.f64,
    numerical_stability_epsilon: ti.f64,
) -> ti.f64:
    rl_eps = re_lambda + numerical_stability_epsilon
    return (7.0 + 11.0 / rl_eps) / (1.0 + 205.0 / rl_eps)

@ti.kernel
def kget_normalized_accel_variance_ao2008(
    re_lambda: ti.types.ndarray(dtype=ti.f64, ndim=1),
    numerical_stability_epsilon: ti.f64,
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_normalized_accel_variance_ao2008(
            re_lambda[i], numerical_stability_epsilon
        )

@register("get_normalized_accel_variance_ao2008", backend="taichi")
def get_normalized_accel_variance_ao2008_taichi(
    re_lambda,
    numerical_stability_epsilon: float = 1e-14,
):
    if not isinstance(numerical_stability_epsilon, (float, int)):
        raise TypeError("ε must be a float or int.")
    rl_np = np.atleast_1d(re_lambda).astype(np.float64)
    n = rl_np.size
    rl_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    rl_ti.from_numpy(rl_np)
    kget_normalized_accel_variance_ao2008(
        rl_ti, float(numerical_stability_epsilon), res_ti
    )
    out = res_ti.to_numpy()
    return out.item() if out.size == 1 else out
