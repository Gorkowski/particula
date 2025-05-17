"""Taichi version of mass_transfer.py."""

import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register
from particula.util.constants import GAS_CONSTANT

from particula.backend.taichi.gas.properties.ti_mean_free_path_module import (
    fget_molecule_mean_free_path,
)
from particula.backend.taichi.particles.properties.ti_knudsen_number_module import (
    fget_knudsen_number,
)
from particula.backend.taichi.particles.properties.ti_vapor_correction_module import (
    fget_vapor_transition_correction,
)

PI = np.pi
GAS_R = float(GAS_CONSTANT)


@ti.func
def fget_first_order_mass_transport_k(
    r: ti.f64, vt: ti.f64, d: ti.f64
) -> ti.f64:
    return 4.0 * PI * r * d * vt


@ti.func
def fget_mass_transfer_rate(
    dp: ti.f64, k: ti.f64, t: ti.f64, m: ti.f64
) -> ti.f64:
    return k * dp * m / (ti.static(GAS_R) * t)


@ti.func
def fget_radius_transfer_rate(dm: ti.f64, r: ti.f64, rho: ti.f64) -> ti.f64:
    return dm / (rho * 4.0 * PI * r * r)


@ti.func
def fget_first_order_mass_transport_via_system_state(  # r × species → k_ij
    particle_radius: ti.f64,
    molar_mass: ti.f64,
    accommodation: ti.f64,
    temperature: ti.f64,
    pressure: ti.f64,
    dynamic_viscosity: ti.f64,
    diffusion_coefficient: ti.f64,
) -> ti.f64:
    mean_free_path = fget_molecule_mean_free_path(
        molar_mass, temperature, pressure, dynamic_viscosity
    )
    kn = fget_knudsen_number(mean_free_path, particle_radius)
    vt = fget_vapor_transition_correction(kn, accommodation)
    return fget_first_order_mass_transport_k(
        particle_radius, vt, diffusion_coefficient
    )


@ti.kernel
def kget_first_order_mass_transport_k(
    r: ti.types.ndarray(dtype=ti.f64, ndim=1),
    vt: ti.types.ndarray(dtype=ti.f64, ndim=1),
    d: ti.types.ndarray(dtype=ti.f64, ndim=1),
    res: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(res.shape[0]):
        res[i] = fget_first_order_mass_transport_k(r[i], vt[i], d[i])


@ti.kernel
def kget_mass_transfer_rate(
    dp: ti.types.ndarray(dtype=ti.f64, ndim=1),
    k: ti.types.ndarray(dtype=ti.f64, ndim=1),
    t: ti.types.ndarray(dtype=ti.f64, ndim=1),
    m: ti.types.ndarray(dtype=ti.f64, ndim=1),
    res: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(res.shape[0]):
        res[i] = fget_mass_transfer_rate(dp[i], k[i], t[i], m[i])


@ti.kernel
def kget_radius_transfer_rate(
    dm: ti.types.ndarray(dtype=ti.f64, ndim=1),
    r: ti.types.ndarray(dtype=ti.f64, ndim=1),
    rho: ti.types.ndarray(dtype=ti.f64, ndim=1),
    res: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(res.shape[0]):
        res[i] = fget_radius_transfer_rate(dm[i], r[i], rho[i])


@ti.kernel
def kget_first_order_mass_transport_via_system_state(
    particle_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    molar_mass: ti.types.ndarray(dtype=ti.f64, ndim=1),
    accommodation: ti.types.ndarray(dtype=ti.f64, ndim=1),
    temperature: ti.f64,
    pressure: ti.f64,
    dynamic_viscosity: ti.f64,
    diffusion_coefficient: ti.f64,
    result: ti.types.ndarray(dtype=ti.f64, ndim=2),
):
    for i in range(particle_radius.shape[0]):  # particles
        for j in range(molar_mass.shape[0]):  # species
            result[i, j] = fget_first_order_mass_transport_via_system_state(
                particle_radius[i],
                molar_mass[j],
                accommodation[i],
                temperature,
                pressure,
                dynamic_viscosity,
                diffusion_coefficient,
            )


@register("get_first_order_mass_transport_k", backend="taichi")
def ti_get_first_order_mass_transport_k(
    particle_radius, vapor_transition, diffusion_coefficient=2e-5
):
    """Taichi version of get_first_order_mass_transport_k."""
    import numpy as np

    if not (
        isinstance(particle_radius, np.ndarray)
        and isinstance(vapor_transition, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")
    a1 = np.atleast_1d(particle_radius).astype(np.float64)
    a2 = np.atleast_1d(vapor_transition).astype(np.float64)
    n = a1.size
    # Broadcast diffusion_coefficient to match input size
    if np.isscalar(diffusion_coefficient):
        d = np.full(n, diffusion_coefficient, dtype=np.float64)
    else:
        d = np.atleast_1d(diffusion_coefficient).astype(np.float64)
        if d.size != n:
            d = np.broadcast_to(d, (n,))
    r_ti = ti.ndarray(dtype=ti.f64, shape=n)
    vt_ti = ti.ndarray(dtype=ti.f64, shape=n)
    d_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    r_ti.from_numpy(a1)
    vt_ti.from_numpy(a2)
    d_ti.from_numpy(d)
    kget_first_order_mass_transport_k(r_ti, vt_ti, d_ti, res_ti)
    result_np = res_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np


@register("get_mass_transfer_rate", backend="taichi")
def ti_get_mass_transfer_rate(
    pressure_delta, first_order_mass_transport, temperature, molar_mass
):
    """Taichi version of get_mass_transfer_rate."""
    import numpy as np

    if not (
        isinstance(pressure_delta, np.ndarray)
        and isinstance(first_order_mass_transport, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")
    dp = np.atleast_1d(pressure_delta).astype(np.float64)
    k = np.atleast_1d(first_order_mass_transport).astype(np.float64)
    t = np.atleast_1d(temperature).astype(np.float64)
    m = np.atleast_1d(molar_mass).astype(np.float64)
    n = dp.size
    dp_ti = ti.ndarray(dtype=ti.f64, shape=n)
    k_ti = ti.ndarray(dtype=ti.f64, shape=n)
    t_ti = ti.ndarray(dtype=ti.f64, shape=n)
    m_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    dp_ti.from_numpy(dp)
    k_ti.from_numpy(k)
    t_ti.from_numpy(t if t.size == n else np.full(n, t[0], dtype=np.float64))
    m_ti.from_numpy(m if m.size == n else np.full(n, m[0], dtype=np.float64))
    kget_mass_transfer_rate(dp_ti, k_ti, t_ti, m_ti, res_ti)
    result_np = res_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np


@register("get_radius_transfer_rate", backend="taichi")
def ti_get_radius_transfer_rate(mass_rate, particle_radius, density):
    """Taichi version of get_radius_transfer_rate."""
    import numpy as np

    if not (
        isinstance(mass_rate, np.ndarray)
        and isinstance(particle_radius, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")
    dm = np.atleast_1d(mass_rate).astype(np.float64)
    r = np.atleast_1d(particle_radius).astype(np.float64)
    rho = np.atleast_1d(density).astype(np.float64)
    n = dm.size
    dm_ti = ti.ndarray(dtype=ti.f64, shape=n)
    r_ti = ti.ndarray(dtype=ti.f64, shape=n)
    rho_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    dm_ti.from_numpy(dm)
    r_ti.from_numpy(r)
    rho_ti.from_numpy(
        rho if rho.size == n else np.full(n, rho[0], dtype=np.float64)
    )
    kget_radius_transfer_rate(dm_ti, r_ti, rho_ti, res_ti)
    result_np = res_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np


@register("get_first_order_mass_transport_via_system_state", backend="taichi")
def ti_get_first_order_mass_transport_via_system_state(
    particle_radius,
    molar_mass,
    accommodation_coefficient,
    temperature,
    pressure,
    dynamic_viscosity,
    diffusion_coefficient,
):
    import numpy as np

    r = np.atleast_1d(particle_radius).astype(np.float64)
    mm = np.atleast_1d(molar_mass).astype(np.float64)
    ac = np.atleast_1d(accommodation_coefficient).astype(np.float64)
    if ac.size != r.size:
        raise ValueError(
            "accommodation_coefficient must match particle_radius length"
        )

    np_particles, np_species = r.size, mm.size
    r_ti = ti.ndarray(dtype=ti.f64, shape=np_particles)
    mm_ti = ti.ndarray(dtype=ti.f64, shape=np_species)
    ac_ti = ti.ndarray(dtype=ti.f64, shape=np_particles)
    res_ti = ti.ndarray(dtype=ti.f64, shape=(np_particles, np_species))

    r_ti.from_numpy(r)
    mm_ti.from_numpy(mm)
    ac_ti.from_numpy(ac)

    kget_first_order_mass_transport_via_system_state(
        r_ti,
        mm_ti,
        ac_ti,
        float(temperature),
        float(pressure),
        float(dynamic_viscosity),
        float(diffusion_coefficient),
        res_ti,
    )
    return res_ti.to_numpy()
