"""Taichi-accelerated Debye function (special_functions.get_debye_function)."""
import taichi as ti
import numpy as np
from particula.backend import register


# --------------------------------------------------------------------------- #
# 3 – element-wise Taichi function                                            #
# --------------------------------------------------------------------------- #
@ti.func
def fget_debye_function(variable: ti.f64, exponent: ti.f64) -> ti.f64:
    """Generalised Debye function, trapezoid rule with 1000 points."""
    n_points = 1000
    if variable <= 0.0:  # avoid 0/0 when variable == 0
        return 0.0

    dt = variable / (n_points - 1)
    integral = 0.0
    for j in range(1, n_points):                # t = 0 gives 0 → skip
        t = dt * j
        weight = 0.5 if j == n_points - 1 else 1.0  # trap. rule end-point weight
        integral += weight * (t ** exponent) / (ti.exp(t) - 1.0)
    integral *= dt

    return (integral / variable
            if exponent == 1.0
            else (exponent / variable ** exponent) * integral)


# --------------------------------------------------------------------------- #
# 4 – vectorised Taichi kernel                                                #
# --------------------------------------------------------------------------- #
@ti.kernel
def kget_debye_function(                               # noqa: N802
    variable: ti.types.ndarray(dtype=ti.f64, ndim=1),
    exponent: ti.f64,
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_debye_function(variable[i], exponent)


# --------------------------------------------------------------------------- #
# 5 – public wrapper registered for the backend                               #
# --------------------------------------------------------------------------- #
@register("get_debye_function", backend="taichi")       # noqa: D401
def ti_get_debye_function(                              # noqa: N802
    variable,
    integration_points: int = 1000,
    n: int = 1,
):
    """Taichi wrapper replicating particles.properties.special_functions."""
    # only the default grid supported for now
    if integration_points != 1000:
        raise NotImplementedError(
            "Taichi backend currently supports integration_points = 1000 only."
        )

    single_value = np.isscalar(variable)
    arr = np.atleast_1d(variable).astype(np.float64)

    var_ti = ti.ndarray(dtype=ti.f64, shape=arr.size)
    res_ti = ti.ndarray(dtype=ti.f64, shape=arr.size)
    var_ti.from_numpy(arr)

    kget_debye_function(var_ti, float(n), res_ti)
    out = res_ti.to_numpy()

    return out[0] if single_value else out.reshape(np.shape(variable))
