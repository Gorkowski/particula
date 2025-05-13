"""Taichi version of mass_transfer.py."""

import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register
from particula.util.constants import GAS_CONSTANT
from numpy import broadcast_arrays, broadcast_to

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
    # ── convert to np.ndarray ────────────────────────────────────────────────
    r  = np.asarray(particle_radius,   dtype=np.float64)
    vt = np.asarray(vapor_transition,  dtype=np.float64)

    # expand radius to 2-D if vapor_transition is 2-D (python version does)
    if vt.ndim == 2 and r.ndim == 1:
        r = r[:, np.newaxis]

    # broadcast radius and vapor-transition to common shape
    r, vt = broadcast_arrays(r, vt)

    # handle diffusion coefficient (scalar, 1-D or 2-D)
    if np.isscalar(diffusion_coefficient):
        d = np.full_like(r, diffusion_coefficient, dtype=np.float64)
    else:
        d = np.asarray(diffusion_coefficient, dtype=np.float64)
        d  = broadcast_to(d, r.shape)

    # flatten for Taichi kernel
    r_flat  = r.ravel()
    vt_flat = vt.ravel()
    d_flat  = d.ravel()
    n = r_flat.size

    r_ti  = ti.ndarray(dtype=ti.f64, shape=n);   r_ti.from_numpy(r_flat)
    vt_ti = ti.ndarray(dtype=ti.f64, shape=n);  vt_ti.from_numpy(vt_flat)
    d_ti  = ti.ndarray(dtype=ti.f64, shape=n);   d_ti.from_numpy(d_flat)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    kget_first_order_mass_transport_k(r_ti, vt_ti, d_ti, res_ti)

    return res_ti.to_numpy().reshape(r.shape).item() \
           if r.size == 1 else res_ti.to_numpy().reshape(r.shape)

@register("get_mass_transfer_rate", backend="taichi")
def ti_get_mass_transfer_rate(
    pressure_delta, first_order_mass_transport, temperature, molar_mass
):
    """Taichi version of get_mass_transfer_rate."""
    import numpy as np
    dp = np.asarray(pressure_delta,           dtype=np.float64)
    k  = np.asarray(first_order_mass_transport, dtype=np.float64)

    # broadcast primary operands
    dp, k = broadcast_arrays(dp, k)

    t = np.asarray(temperature, dtype=np.float64)
    m = np.asarray(molar_mass, dtype=np.float64)

    # broadcast scalars / vectors to dp,k shape
    if t.size == 1:
        t = np.full(dp.shape, t.item(), dtype=np.float64)
    else:
        t = broadcast_to(t, dp.shape)

    if m.size == 1:
        m = np.full(dp.shape, m.item(), dtype=np.float64)
    else:
        m = broadcast_to(m, dp.shape)

    # flatten
    dp_f, k_f, t_f, m_f = [arr.ravel() for arr in (dp, k, t, m)]
    n = dp_f.size

    dp_ti = ti.ndarray(dtype=ti.f64, shape=n); dp_ti.from_numpy(dp_f)
    k_ti  = ti.ndarray(dtype=ti.f64, shape=n);  k_ti.from_numpy(k_f)
    t_ti  = ti.ndarray(dtype=ti.f64, shape=n);  t_ti.from_numpy(t_f)
    m_ti  = ti.ndarray(dtype=ti.f64, shape=n);  m_ti.from_numpy(m_f)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    kget_mass_transfer_rate(dp_ti, k_ti, t_ti, m_ti, res_ti)

    return res_ti.to_numpy().reshape(dp.shape).item() \
           if dp.size == 1 else res_ti.to_numpy().reshape(dp.shape)

@register("get_radius_transfer_rate", backend="taichi")
def ti_get_radius_transfer_rate(mass_rate, particle_radius, density):
    """Taichi version of get_radius_transfer_rate."""
    import numpy as np
    dm  = np.asarray(mass_rate,        dtype=np.float64)
    r   = np.asarray(particle_radius,  dtype=np.float64)
    rho = np.asarray(density,          dtype=np.float64)

    # if mass_rate is 2-D make radius 2-D in the first axis
    if dm.ndim == 2 and r.ndim == 1:
        r = r[:, np.newaxis]

    dm, r = broadcast_arrays(dm, r)

    # density broadcast
    if rho.size == 1:
        rho = np.full(dm.shape, rho.item(), dtype=np.float64)
    else:
        if rho.ndim == 1 and dm.ndim == 2:
            rho = rho[:, np.newaxis]
        rho = broadcast_to(rho, dm.shape)

    # flatten
    dm_f, r_f, rho_f = [arr.ravel() for arr in (dm, r, rho)]
    n = dm_f.size

    dm_ti  = ti.ndarray(dtype=ti.f64, shape=n);  dm_ti.from_numpy(dm_f)
    r_ti   = ti.ndarray(dtype=ti.f64, shape=n);   r_ti.from_numpy(r_f)
    rho_ti = ti.ndarray(dtype=ti.f64, shape=n); rho_ti.from_numpy(rho_f)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    kget_radius_transfer_rate(dm_ti, r_ti, rho_ti, res_ti)

    return res_ti.to_numpy().reshape(dm.shape).item() \
           if dm.size == 1 else res_ti.to_numpy().reshape(dm.shape)
