import taichi as ti
import numpy as np

from particula.backend.dispatch_register import register
from particula.util.constants import GAS_CONSTANT

_R = float(GAS_CONSTANT)  # J mol⁻¹ K⁻¹

@ti.func
def fget_molecule_mean_free_path(
    molar_mass: ti.f64,
    temperature: ti.f64,
    pressure: ti.f64,
    dynamic_viscosity: ti.f64,
) -> ti.f64:
    return (2.0 * dynamic_viscosity / pressure) / ti.sqrt(
        8.0 * molar_mass / (ti.pi64 * _R * temperature)
    )

@ti.kernel
def kget_molecule_mean_free_path(          # noqa: N802
    molar_mass: ti.types.ndarray(dtype=ti.f64, ndim=1),
    temperature: ti.types.ndarray(dtype=ti.f64, ndim=1),
    pressure: ti.types.ndarray(dtype=ti.f64, ndim=1),
    dynamic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_molecule_mean_free_path(
            molar_mass[i],
            temperature[i],
            pressure[i],
            dynamic_viscosity[i],
        )

@register("get_molecule_mean_free_path", backend="taichi")
def get_molecule_mean_free_path_taichi(
    molar_mass,
    temperature,
    pressure,
    dynamic_viscosity=None,
):
    if not all(
        isinstance(arg, (float, np.ndarray))
        for arg in (molar_mass, temperature, pressure)
    ):
        raise TypeError("Taichi backend expects float or NumPy array inputs.")

    # default μ if None
    if dynamic_viscosity is None:
        dynamic_viscosity = get_dynamic_viscosity(temperature)

    mm, T, P, mu = map(np.atleast_1d, (molar_mass, temperature, pressure, dynamic_viscosity))
    n = mm.size

    mm_ti  = ti.ndarray(dtype=ti.f64, shape=n);   mm_ti.from_numpy(mm)
    T_ti   = ti.ndarray(dtype=ti.f64, shape=n);   T_ti.from_numpy(T)
    P_ti   = ti.ndarray(dtype=ti.f64, shape=n);   P_ti.from_numpy(P)
    mu_ti  = ti.ndarray(dtype=ti.f64, shape=n);   mu_ti.from_numpy(mu)
    out_ti = ti.ndarray(dtype=ti.f64, shape=n)

    kget_molecule_mean_free_path(mm_ti, T_ti, P_ti, mu_ti, out_ti)

    result_np = out_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np
