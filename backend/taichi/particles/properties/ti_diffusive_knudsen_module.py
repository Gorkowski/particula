"""Taichi implementation of the diffusive Knudsen number."""

import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register
from particula.util.constants import BOLTZMANN_CONSTANT
from particula.util.reduced_quantity import get_reduced_self_broadcast
from particula.particles.properties import coulomb_enhancement as ce


@ti.func
def fget_diffusive_knudsen_number(
    sum_r: ti.f64,
    red_mass: ti.f64,
    red_fric: ti.f64,
    cont_enh: ti.f64,
    kin_enh: ti.f64,
    temp: ti.f64,
) -> ti.f64:
    numerator   = ti.sqrt(temp * BOLTZMANN_CONSTANT * red_mass) / red_fric
    denominator = sum_r * cont_enh / kin_enh
    return numerator / denominator


@ti.kernel
def kget_diffusive_knudsen_number(
    sum_r: ti.types.ndarray(dtype=ti.f64, ndim=2),
    red_mass: ti.types.ndarray(dtype=ti.f64, ndim=2),
    red_fric: ti.types.ndarray(dtype=ti.f64, ndim=2),
    cont_enh: ti.types.ndarray(dtype=ti.f64, ndim=2),
    kin_enh: ti.types.ndarray(dtype=ti.f64, ndim=2),
    temperature: ti.f64,
    result: ti.types.ndarray(dtype=ti.f64, ndim=2),
):
    for i, j in ti.ndrange(result.shape[0], result.shape[1]):
        result[i, j] = fget_diffusive_knudsen_number(
            sum_r[i, j],
            red_mass[i, j],
            red_fric[i, j],
            cont_enh[i, j],
            kin_enh[i, j],
            temperature,
        )


@register("get_diffusive_knudsen_number", backend="taichi")
def ti_get_diffusive_knudsen_number(
    particle_radius,
    particle_mass,
    friction_factor,
    coulomb_potential_ratio=0.0,
    temperature=298.15,
):
    # 4 a  – type guard / promote to 1-D arrays
    for arg in (
        particle_radius,
        particle_mass,
        friction_factor,
        coulomb_potential_ratio,
    ):
        if not isinstance(arg, (np.ndarray, float, int)):
            raise TypeError("Inputs must be float/int or NumPy array.")
    r_arr = np.atleast_1d(particle_radius).astype(np.float64)
    m_arr = np.atleast_1d(particle_mass).astype(np.float64)
    f_arr = np.atleast_1d(friction_factor).astype(np.float64)

    n = r_arr.size
    shape_2d = (n, n)

    # 4 b  – auxiliary NumPy matrices (done on CPU)
    sum_r = r_arr[:, None] + r_arr                       # (n,n)
    red_m = get_reduced_self_broadcast(m_arr)            # (n,n)
    red_f = get_reduced_self_broadcast(f_arr)            # (n,n)

    kin_enh = ce.get_coulomb_kinetic_limit(coulomb_potential_ratio)
    cont_enh = ce.get_coulomb_continuum_limit(coulomb_potential_ratio)

    # broadcast enhancements to (n,n)
    kin_mat  = np.broadcast_to(np.atleast_2d(kin_enh), shape_2d).astype(np.float64)
    cont_mat = np.broadcast_to(np.atleast_2d(cont_enh), shape_2d).astype(np.float64)

    # 4 c  – allocate / fill Taichi NDArrays
    buf_sum   = ti.ndarray(ti.f64, shape=shape_2d); buf_sum.from_numpy(sum_r)
    buf_m     = ti.ndarray(ti.f64, shape=shape_2d); buf_m.from_numpy(red_m)
    buf_f     = ti.ndarray(ti.f64, shape=shape_2d); buf_f.from_numpy(red_f)
    buf_cont  = ti.ndarray(ti.f64, shape=shape_2d); buf_cont.from_numpy(cont_mat)
    buf_kin   = ti.ndarray(ti.f64, shape=shape_2d); buf_kin.from_numpy(kin_mat)
    buf_res   = ti.ndarray(ti.f64, shape=shape_2d)

    # 4 d  – launch kernel
    kget_diffusive_knudsen_number(
        buf_sum, buf_m, buf_f, buf_cont, buf_kin, float(temperature), buf_res
    )

    res = buf_res.to_numpy()
    return res.item() if res.size == 1 else res
