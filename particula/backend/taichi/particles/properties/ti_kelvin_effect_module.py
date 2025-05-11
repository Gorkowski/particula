# 1 – imports
import taichi as ti
import numpy as np

from particula.backend.dispatch_register import register
from particula.util.constants import GAS_CONSTANT

# 2 – python-side constants (used inside ti.func with ti.static)
_EXP_MAX = np.log(np.finfo(np.float64).max)        # ≈ 7.097e2

# 3 – element-wise taichi funcs
@ti.func
def fget_kelvin_radius(σ: ti.f64, ρ: ti.f64, M: ti.f64, T: ti.f64) -> ti.f64:
    R = ti.static(GAS_CONSTANT)
    return (2.0 * σ * M) / (R * T * ρ)


@ti.func
def fget_kelvin_term(r_p: ti.f64, r_k: ti.f64) -> ti.f64:
    expo = r_k / r_p
    max_exp = ti.static(_EXP_MAX)
    expo = ti.min(expo, max_exp)    # overflow protection
    return ti.exp(expo)

# 4 – kernels
@ti.kernel
def kget_kelvin_radius(
    σ: ti.types.ndarray(dtype=ti.f64, ndim=1),
    ρ: ti.types.ndarray(dtype=ti.f64, ndim=1),
    M: ti.types.ndarray(dtype=ti.f64, ndim=1),
    T: ti.f64,
    res: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(res.shape[0]):
        res[i] = fget_kelvin_radius(σ[i], ρ[i], M[i], T)


@ti.kernel
def kget_kelvin_term(
    r_p: ti.types.ndarray(dtype=ti.f64, ndim=2),
    r_k: ti.types.ndarray(dtype=ti.f64, ndim=2),
    res: ti.types.ndarray(dtype=ti.f64, ndim=2),
):
    for I in ti.grouped(res):
        res[I] = fget_kelvin_term(r_p[I], r_k[I])

# 5 – public wrappers with backend registration
@register("get_kelvin_radius", backend="taichi")
def ti_get_kelvin_radius(surface_tension, density, molar_mass, temperature):
    if not all(isinstance(x, np.ndarray) for x in (surface_tension, density, molar_mass)):
        raise TypeError("Taichi backend expects NumPy arrays for first three inputs.")
    if not np.isscalar(temperature):
        raise TypeError("Temperature must be a scalar.")

    σ, ρ, M = map(np.atleast_1d, (surface_tension, density, molar_mass))
    n = σ.size
    σ_ti, ρ_ti, M_ti = (ti.ndarray(dtype=ti.f64, shape=n) for _ in range(3))
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    σ_ti.from_numpy(σ)
    ρ_ti.from_numpy(ρ)
    M_ti.from_numpy(M)

    kget_kelvin_radius(σ_ti, ρ_ti, M_ti, float(temperature), res_ti)
    out = res_ti.to_numpy()
    return out.item() if out.size == 1 else out


@register("get_kelvin_term", backend="taichi")
def ti_get_kelvin_term(particle_radius, kelvin_radius_value):
    if not (isinstance(particle_radius, np.ndarray)
            and isinstance(kelvin_radius_value, np.ndarray)):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")

    pr = particle_radius.astype(np.float64).ravel()
    kr = kelvin_radius_value.astype(np.float64).ravel()

    # replicate NumPy implementation: outer product only if both arrays > 1
    expand = (kr.size > 1) and (pr.size > 1)

    if expand:                                        # outer-product case
        pr_mat = np.broadcast_to(pr[:, None], (pr.size, kr.size))
        kr_mat = np.broadcast_to(kr[None, :], (pr.size, kr.size))
    else:                                             # no expansion
        pr_mat = pr[:, None]                          # (n, 1)
        kr_mat = np.broadcast_to(kr, pr_mat.shape)    # (n, 1)

    shape_2d = pr_mat.shape                           # (rows, cols)
    pr_ti = ti.ndarray(ti.f64, shape_2d);  pr_ti.from_numpy(pr_mat)
    kr_ti = ti.ndarray(ti.f64, shape_2d);  kr_ti.from_numpy(kr_mat)
    res_ti = ti.ndarray(ti.f64, shape_2d)

    kget_kelvin_term(pr_ti, kr_ti, res_ti)
    out = res_ti.to_numpy()

    if not expand:                                    # restore 1-D result
        out = out[:, 0]
    return out.item() if out.size == 1 else out
