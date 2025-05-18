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
    particle_radius: ti.f64,
    vapor_transition: ti.f64,
    diffusion_coefficient: ti.f64,
) -> ti.f64:
    return 4.0 * PI * particle_radius * diffusion_coefficient * vapor_transition


@ti.func
def fget_mass_transfer_rate(
    pressure_delta: ti.f64,
    first_order_mass_transport: ti.f64,
    temperature: ti.f64,
    molar_mass: ti.f64,
) -> ti.f64:
    return (
        first_order_mass_transport
        * pressure_delta
        * molar_mass
        / (ti.static(GAS_R) * temperature)
    )


@ti.func
def fget_radius_transfer_rate(
    mass_rate: ti.f64,
    particle_radius: ti.f64,
    density: ti.f64,
) -> ti.f64:
    return mass_rate / (density * 4.0 * PI * particle_radius * particle_radius)


@ti.func
def fget_first_order_mass_transport_via_system_state(  # r × species → k_ij
    particle_radius: ti.f64,
    molar_mass: ti.f64,
    mass_accommodation: ti.f64,
    temperature: ti.f64,
    pressure: ti.f64,
    dynamic_viscosity: ti.f64,
    diffusion_coefficient: ti.f64,
) -> ti.f64:
    mean_free_path = fget_molecule_mean_free_path(
        molar_mass, temperature, pressure, dynamic_viscosity
    )
    kn = fget_knudsen_number(mean_free_path, particle_radius)
    vt = fget_vapor_transition_correction(kn, mass_accommodation)
    return fget_first_order_mass_transport_k(
        particle_radius, vt, diffusion_coefficient
    )


@ti.kernel
def kget_first_order_mass_transport_k(
    particle_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    vapor_transition: ti.types.ndarray(dtype=ti.f64, ndim=1),
    diffusion_coefficient: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_first_order_mass_transport_k(
            particle_radius[i], vapor_transition[i], diffusion_coefficient[i]
        )


@ti.kernel
def kget_mass_transfer_rate(
    pressure_delta: ti.types.ndarray(dtype=ti.f64, ndim=1),
    first_order_mass_transport: ti.types.ndarray(dtype=ti.f64, ndim=1),
    temperature: ti.types.ndarray(dtype=ti.f64, ndim=1),
    molar_mass: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_mass_transfer_rate(
            pressure_delta[i], first_order_mass_transport[i], temperature[i], molar_mass[i]
        )


@ti.kernel
def kget_radius_transfer_rate(
    mass_rate: ti.types.ndarray(dtype=ti.f64, ndim=1),
    particle_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    density: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_radius_transfer_rate(
            mass_rate[i], particle_radius[i], density[i]
        )


@ti.kernel
def kget_first_order_mass_transport_via_system_state(
    particle_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    molar_mass: ti.types.ndarray(dtype=ti.f64, ndim=1),
    mass_accommodation: ti.types.ndarray(dtype=ti.f64, ndim=1),
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
                mass_accommodation[i],
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
    particle_radius_array = np.atleast_1d(particle_radius).astype(np.float64)
    vapor_transition_array = np.atleast_1d(vapor_transition).astype(np.float64)
    n_points = particle_radius_array.size
    # Broadcast diffusion_coefficient to match input size
    if np.isscalar(diffusion_coefficient):
        diffusion_coefficient_array = np.full(n_points, diffusion_coefficient, dtype=np.float64)
    else:
        diffusion_coefficient_array = np.atleast_1d(diffusion_coefficient).astype(np.float64)
        if diffusion_coefficient_array.size != n_points:
            diffusion_coefficient_array = np.broadcast_to(diffusion_coefficient_array, (n_points,))
    particle_radius_ti = ti.ndarray(dtype=ti.f64, shape=n_points)
    vapor_transition_ti = ti.ndarray(dtype=ti.f64, shape=n_points)
    diffusion_coefficient_ti = ti.ndarray(dtype=ti.f64, shape=n_points)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n_points)
    particle_radius_ti.from_numpy(particle_radius_array)
    vapor_transition_ti.from_numpy(vapor_transition_array)
    diffusion_coefficient_ti.from_numpy(diffusion_coefficient_array)
    kget_first_order_mass_transport_k(
        particle_radius_ti, vapor_transition_ti, diffusion_coefficient_ti, result_ti
    )
    result_np = result_ti.to_numpy()
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
    pressure_delta_array = np.atleast_1d(pressure_delta).astype(np.float64)
    first_order_mass_transport_array = np.atleast_1d(first_order_mass_transport).astype(np.float64)
    temperature_array = np.atleast_1d(temperature).astype(np.float64)
    molar_mass_array = np.atleast_1d(molar_mass).astype(np.float64)
    n_points = pressure_delta_array.size
    pressure_delta_ti = ti.ndarray(dtype=ti.f64, shape=n_points)
    first_order_mass_transport_ti = ti.ndarray(dtype=ti.f64, shape=n_points)
    temperature_ti = ti.ndarray(dtype=ti.f64, shape=n_points)
    molar_mass_ti = ti.ndarray(dtype=ti.f64, shape=n_points)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n_points)
    pressure_delta_ti.from_numpy(pressure_delta_array)
    first_order_mass_transport_ti.from_numpy(first_order_mass_transport_array)
    temperature_ti.from_numpy(
        temperature_array if temperature_array.size == n_points else np.full(n_points, temperature_array[0], dtype=np.float64)
    )
    molar_mass_ti.from_numpy(
        molar_mass_array if molar_mass_array.size == n_points else np.full(n_points, molar_mass_array[0], dtype=np.float64)
    )
    kget_mass_transfer_rate(
        pressure_delta_ti, first_order_mass_transport_ti, temperature_ti, molar_mass_ti, result_ti
    )
    result_np = result_ti.to_numpy()
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
    mass_rate_array = np.atleast_1d(mass_rate).astype(np.float64)
    particle_radius_array = np.atleast_1d(particle_radius).astype(np.float64)
    density_array = np.atleast_1d(density).astype(np.float64)
    n_points = mass_rate_array.size
    mass_rate_ti = ti.ndarray(dtype=ti.f64, shape=n_points)
    particle_radius_ti = ti.ndarray(dtype=ti.f64, shape=n_points)
    density_ti = ti.ndarray(dtype=ti.f64, shape=n_points)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n_points)
    mass_rate_ti.from_numpy(mass_rate_array)
    particle_radius_ti.from_numpy(particle_radius_array)
    density_ti.from_numpy(
        density_array if density_array.size == n_points else np.full(n_points, density_array[0], dtype=np.float64)
    )
    kget_radius_transfer_rate(
        mass_rate_ti, particle_radius_ti, density_ti, result_ti
    )
    result_np = result_ti.to_numpy()
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

    particle_radius_array = np.atleast_1d(particle_radius).astype(np.float64)
    molar_mass_array = np.atleast_1d(molar_mass).astype(np.float64)
    accommodation_coefficient_array = np.atleast_1d(accommodation_coefficient).astype(np.float64)
    if accommodation_coefficient_array.size != particle_radius_array.size:
        raise ValueError(
            "accommodation_coefficient must match particle_radius length"
        )

    n_particles, n_species = particle_radius_array.size, molar_mass_array.size
    particle_radius_ti = ti.ndarray(dtype=ti.f64, shape=n_particles)
    molar_mass_ti = ti.ndarray(dtype=ti.f64, shape=n_species)
    mass_accommodation_ti = ti.ndarray(dtype=ti.f64, shape=n_particles)
    result_ti = ti.ndarray(dtype=ti.f64, shape=(n_particles, n_species))

    particle_radius_ti.from_numpy(particle_radius_array)
    molar_mass_ti.from_numpy(molar_mass_array)
    mass_accommodation_ti.from_numpy(accommodation_coefficient_array)

    kget_first_order_mass_transport_via_system_state(
        particle_radius_ti,
        molar_mass_ti,
        mass_accommodation_ti,
        float(temperature),
        float(pressure),
        float(dynamic_viscosity),
        float(diffusion_coefficient),
        result_ti,
    )
    return result_ti.to_numpy()
