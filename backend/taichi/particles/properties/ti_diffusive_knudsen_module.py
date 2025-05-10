"""Taichi implementation of the diffusive Knudsen number."""

import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register
from particula.util.constants import BOLTZMANN_CONSTANT
from particula.backend.taichi.particles.properties import ti_coulomb_enhancement_module as ti_ce


@ti.func
def fget_diffusive_knudsen_number(
    radius_i: ti.f64,
    radius_j: ti.f64,
    mass_i: ti.f64,
    mass_j: ti.f64,
    friction_i: ti.f64,
    friction_j: ti.f64,
    potential_i: ti.f64,
    potential_j: ti.f64,
    temperature: ti.f64,
) -> ti.f64:
    sum_of_radii = radius_i + radius_j
    reduced_mass = (mass_i * mass_j) / (mass_i + mass_j)
    reduced_friction = (friction_i * friction_j) / (friction_i + friction_j)
    kin_enhance = ti_ce.fget_coulomb_kinetic_limit(0.5 * (potential_i + potential_j))
    con_enhance = ti_ce.fget_coulomb_continuum_limit(0.5 * (potential_i + potential_j))
    numerator = ti.sqrt(temperature * BOLTZMANN_CONSTANT * reduced_mass) / reduced_friction
    denominator = sum_of_radii * con_enhance / kin_enhance
    return numerator / denominator


@ti.kernel
def kget_diffusive_knudsen_number(
    radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    mass: ti.types.ndarray(dtype=ti.f64, ndim=1),
    friction: ti.types.ndarray(dtype=ti.f64, ndim=1),
    potential: ti.types.ndarray(dtype=ti.f64, ndim=1),
    temperature: ti.f64,
    result: ti.types.ndarray(dtype=ti.f64, ndim=2),
):
    for i, j in ti.ndrange(result.shape[0], result.shape[1]):
        result[i, j] = fget_diffusive_knudsen_number(
            radius[i], radius[j],
            mass[i], mass[j],
            friction[i], friction[j],
            potential[i], potential[j],
            temperature,
        )


@register("get_diffusive_knudsen_number", backend="taichi")
def ti_get_diffusive_knudsen_number(
    particle_radius, particle_mass, friction_factor,
    coulomb_potential_ratio=0.0, temperature=298.15
):
    """
    Taichi implementation of get_diffusive_knudsen_number.
    Behaviour & signature identical to the NumPy version.
    """
    if not all(isinstance(x, np.ndarray) for x in
               [np.atleast_1d(particle_radius),
                np.atleast_1d(particle_mass),
                np.atleast_1d(friction_factor)]):
        raise TypeError("Taichi backend expects NumPy arrays for radius, mass and friction.")

    r = np.atleast_1d(particle_radius).astype(np.float64)
    m = np.atleast_1d(particle_mass).astype(np.float64)
    f = np.atleast_1d(friction_factor).astype(np.float64)
    cp = np.atleast_1d(coulomb_potential_ratio).astype(np.float64)

    if not (r.size == m.size == f.size == cp.size):
        raise ValueError("All 1-D inputs must have the same length.")

    n = r.size

    r_ti = ti.ndarray(dtype=ti.f64, shape=n)
    m_ti = ti.ndarray(dtype=ti.f64, shape=n)
    f_ti = ti.ndarray(dtype=ti.f64, shape=n)
    cp_ti = ti.ndarray(dtype=ti.f64, shape=n)
    out_ti = ti.ndarray(dtype=ti.f64, shape=(n, n))

    r_ti.from_numpy(r)
    m_ti.from_numpy(m)
    f_ti.from_numpy(f)
    cp_ti.from_numpy(cp)

    kget_diffusive_knudsen_number(r_ti, m_ti, f_ti, cp_ti, temperature, out_ti)

    result_np = out_ti.to_numpy()
    if result_np.size == 1:
        return result_np.item()
    return result_np
