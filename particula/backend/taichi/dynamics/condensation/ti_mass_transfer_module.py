"""Taichi version of mass_transfer.py."""

import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register
from particula.util.constants import GAS_CONSTANT

PI = np.pi
GAS_R = float(GAS_CONSTANT)

@ti.func
def fget_first_order_mass_transport_k(r: ti.f64, vt: ti.f64, d: ti.f64) -> ti.f64:
    return 4.0 * PI * r * d * vt

@ti.func
def fget_mass_transfer_rate(dp: ti.f64, k: ti.f64, t: ti.f64, m: ti.f64) -> ti.f64:
    return k * dp * m / (ti.static(GAS_R) * t)

@ti.func
def fget_radius_transfer_rate(dm: ti.f64, r: ti.f64, rho: ti.f64) -> ti.f64:
    return dm / (rho * 4.0 * PI * r * r)

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

@register("get_first_order_mass_transport_k", backend="taichi")
def ti_get_first_order_mass_transport_k(
    particle_radius, vapor_transition, diffusion_coefficient=2e-5
):
    """Taichi version of get_first_order_mass_transport_k."""
    import numpy as np
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
def ti_get_radius_transfer_rate(
    mass_rate, particle_radius, density
):
    """Taichi version of get_radius_transfer_rate."""
    import numpy as np
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
    rho_ti.from_numpy(rho if rho.size == n else np.full(n, rho[0], dtype=np.float64))
    kget_radius_transfer_rate(dm_ti, r_ti, rho_ti, res_ti)
    result_np = res_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np
