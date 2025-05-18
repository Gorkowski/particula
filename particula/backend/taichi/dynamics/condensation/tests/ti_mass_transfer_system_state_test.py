import taichi as ti
import numpy as np
import numpy.testing as npt
ti.init(arch=ti.cpu)

# taichi wrapper + kernel
from particula.backend.taichi.dynamics.condensation.ti_mass_transfer import (
    ti_get_first_order_mass_transport_via_system_state,
    kget_first_order_mass_transport_via_system_state,
)

def _example_inputs():
    particle_radius        = np.array([1e-8, 2e-6], dtype=np.float64)
    molar_mass             = np.array([0.02897, 0.04401], dtype=np.float64)
    mass_accommodation     = np.array([1.0, 1.0], dtype=np.float64)
    temperature            = 300.0
    pressure               = 101325.0
    dynamic_viscosity      = 1.8e-5
    diffusion_coefficient  = 2e-5
    return (
        particle_radius,
        molar_mass,
        mass_accommodation,
        temperature,
        pressure,
        dynamic_viscosity,
        diffusion_coefficient,
    )

def test_wrapper():
    (
        particle_radius,
        molar_mass,
        mass_accommodation,
        temperature,
        pressure,
        dynamic_viscosity,
        diffusion_coefficient,
    ) = _example_inputs()
    expected = np.array(
        [[2.740521e-13, 3.331703e-13], [4.909421e-10, 4.931564e-10]],
        dtype=np.float64,
    )
    result = ti_get_first_order_mass_transport_via_system_state(
        particle_radius,
        molar_mass,
        mass_accommodation,
        temperature,
        pressure,
        dynamic_viscosity,
        diffusion_coefficient,
    )
    npt.assert_allclose(result, expected, rtol=1e-6)

def test_kernel_direct():
    (
        particle_radius,
        molar_mass,
        mass_accommodation,
        temperature,
        pressure,
        dynamic_viscosity,
        diffusion_coefficient,
    ) = _example_inputs()

    n_particles, n_species = particle_radius.size, molar_mass.size

    particle_radius_ti   = ti.ndarray(ti.f64, n_particles)
    molar_mass_ti        = ti.ndarray(ti.f64, n_species)
    mass_accommodation_ti = ti.ndarray(ti.f64, n_particles)
    result_ti            = ti.ndarray(ti.f64, (n_particles, n_species))

    particle_radius_ti.from_numpy(particle_radius)
    molar_mass_ti.from_numpy(molar_mass)
    mass_accommodation_ti.from_numpy(mass_accommodation)
    kget_first_order_mass_transport_via_system_state(
        particle_radius_ti,
        molar_mass_ti,
        mass_accommodation_ti,
        temperature,
        pressure,
        dynamic_viscosity,
        diffusion_coefficient,
        result_ti,
    )

    expected = np.array(
        [[2.740521e-13, 3.331703e-13], [4.909421e-10, 4.931564e-10]],
        dtype=np.float64,
    )
    npt.assert_allclose(result_ti.to_numpy(), expected, rtol=1e-6)
