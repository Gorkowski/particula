"""Taichi-accelerated fluid RMS fluctuation velocity calculation module."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

@ti.func
def fget_fluid_rms_velocity(
    re_lambda: ti.f64,
    kinematic_viscosity: ti.f64,
    turbulent_dissipation: ti.f64
) -> ti.f64:
    """Element-wise Taichi function for fluid RMS fluctuation velocity."""
    kolmogorov_velocity = (kinematic_viscosity * turbulent_dissipation) ** 0.25
    return (re_lambda ** 0.5 * kolmogorov_velocity) / (15.0 ** 0.25)

@ti.kernel
def kget_fluid_rms_velocity(
    re_lambda: ti.types.ndarray(dtype=ti.f64, ndim=1),
    kinematic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    turbulent_dissipation: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Vectorized Taichi kernel for fluid RMS fluctuation velocity."""
    for i in range(result.shape[0]):
        result[i] = fget_fluid_rms_velocity(
            re_lambda[i], kinematic_viscosity[i], turbulent_dissipation[i]
        )

@register("get_fluid_rms_velocity", backend="taichi")
def get_fluid_rms_velocity_taichi(
    re_lambda, kinematic_viscosity, turbulent_dissipation
):
    """Taichi backend wrapper for fluid RMS fluctuation velocity."""
    if not (
        isinstance(re_lambda, np.ndarray)
        and isinstance(kinematic_viscosity, np.ndarray)
        and isinstance(turbulent_dissipation, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for all three inputs.")

    rl = np.atleast_1d(re_lambda)
    kv = np.atleast_1d(kinematic_viscosity)
    td = np.atleast_1d(turbulent_dissipation)
    n = rl.size

    rl_ti = ti.ndarray(dtype=ti.f64, shape=n)
    kv_ti = ti.ndarray(dtype=ti.f64, shape=n)
    td_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)

    rl_ti.from_numpy(rl)
    kv_ti.from_numpy(kv)
    td_ti.from_numpy(td)

    kget_fluid_rms_velocity(rl_ti, kv_ti, td_ti, result_ti)

    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np
