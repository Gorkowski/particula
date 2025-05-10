"""Taichi accelerated kinematic-viscosity helpers."""
import taichi as ti
import numpy as np

from particula.backend.dispatch_register import register
from particula.backend.taichi.gas.properties.ti_dynamic_viscosity_module import (
    ti_get_dynamic_viscosity,
)

# ── 3. element-wise func
@ti.func
def fget_kinematic_viscosity(dynamic_viscosity: ti.f64,
                             fluid_density: ti.f64) -> ti.f64:          # noqa: D401
    return dynamic_viscosity / fluid_density

# ── 4. vectorised kernel
@ti.kernel
def kget_kinematic_viscosity(
    dynamic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    fluid_density:     ti.types.ndarray(dtype=ti.f64, ndim=1),
    result:            ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_kinematic_viscosity(dynamic_viscosity[i],
                                             fluid_density[i])

# ── 5a.  public wrapper
@register("get_kinematic_viscosity", backend="taichi")
def ti_get_kinematic_viscosity(dynamic_viscosity, fluid_density):
    if not (isinstance(dynamic_viscosity, np.ndarray)
            and isinstance(fluid_density, np.ndarray)):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")

    mu, rho = np.atleast_1d(dynamic_viscosity), np.atleast_1d(fluid_density)
    n = mu.size

    mu_ti   = ti.ndarray(dtype=ti.f64, shape=n);  mu_ti.from_numpy(mu)
    rho_ti  = ti.ndarray(dtype=ti.f64, shape=n);  rho_ti.from_numpy(rho)
    res_ti  = ti.ndarray(dtype=ti.f64, shape=n)

    kget_kinematic_viscosity(mu_ti, rho_ti, res_ti)

    res_np = res_ti.to_numpy()
    return res_np.item() if res_np.size == 1 else res_np

# ── 5b.  “via system state” helper
@register("get_kinematic_viscosity_via_system_state", backend="taichi")
def ti_get_kinematic_viscosity_via_system_state(
    temperature,                      # K
    fluid_density,                    # kg m⁻³
    reference_viscosity,              # Pa·s
    reference_temperature,            # K
):
    # obtain μ(T) from Taichi implementation
    mu = ti_get_dynamic_viscosity(
        temperature, reference_viscosity, reference_temperature
    )
    # divide by ρ (NumPy broadcast works)
    return mu / fluid_density
