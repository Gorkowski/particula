import taichi as ti, numpy as np, numpy.testing as npt
ti.init(arch=ti.cpu)

from particula.dynamics.condensation.mass_transfer import (
    get_first_order_mass_transport_k as np_fk,
    get_mass_transfer_rate as np_mr,
    get_radius_transfer_rate as np_rr,
)
from particula.backend.taichi.dynamics.condensation.ti_mass_transfer import (
    ti_get_first_order_mass_transport_k as ti_fk,
    kget_first_order_mass_transport_k,
    ti_get_mass_transfer_rate as ti_mr,
    kget_mass_transfer_rate,
    ti_get_radius_transfer_rate as ti_rr,
    kget_radius_transfer_rate,
)

def test_wrapper_functions():
    particle_radius          = np.array([1e-6, 2e-6], dtype=np.float64)
    vapor_transition         = np.array([0.6, 0.7], dtype=np.float64)
    diffusion_coefficient    = np.array([2e-5, 2e-5], dtype=np.float64)
    vapor_pressure_delta     = np.array([10.0, 15.0], dtype=np.float64)
    first_order_mass_transfer = np_fk(
        particle_radius, vapor_transition, diffusion_coefficient
    )
    temperature   = 300.0
    molar_mass    = 0.02897
    mass_transfer_rate = np_mr(
        vapor_pressure_delta,
        first_order_mass_transfer,
        temperature,
        molar_mass,
    )

    npt.assert_allclose(
        ti_fk(particle_radius, vapor_transition, diffusion_coefficient),
        first_order_mass_transfer,
    )
    npt.assert_allclose(
        ti_mr(vapor_pressure_delta,
              first_order_mass_transfer,
              temperature,
              molar_mass),
        mass_transfer_rate,
    )
    particle_density = 1000.0
    npt.assert_allclose(
        ti_rr(mass_transfer_rate, particle_radius, particle_density),
        np_rr(mass_transfer_rate, particle_radius, particle_density),
    )

def test_kernels_direct():
    particle_radius       = np.array([1e-6, 2e-6], dtype=np.float64)
    vapor_transition      = np.array([0.6, 0.7], dtype=np.float64)
    diffusion_coefficient = np.array([2e-5, 2e-5], dtype=np.float64)
    n_particles           = particle_radius.size
    radius_ti, vapor_transition_ti, diff_coeff_ti = [
        ti.ndarray(ti.f64, n_particles) for _ in range(3)
    ]
    res_ti = ti.ndarray(ti.f64, n_particles)
    radius_ti.from_numpy(particle_radius)
    vapor_transition_ti.from_numpy(vapor_transition)
    diff_coeff_ti.from_numpy(diffusion_coefficient)
    kget_first_order_mass_transport_k(
        radius_ti, vapor_transition_ti, diff_coeff_ti, res_ti
    )
    npt.assert_allclose(
        res_ti.to_numpy(),
        np_fk(particle_radius, vapor_transition, diffusion_coefficient),
    )

    # ----- mass-transfer-rate kernel ---------------------------------------
    vapor_pressure_delta = np.array([10.0, 15.0], dtype=np.float64)
    first_order_mass_transfer = np_fk(
        particle_radius, vapor_transition, diffusion_coefficient
    )
    temperature_t = np.full(vapor_pressure_delta.size, 300.0, dtype=np.float64)
    molar_mass_m  = np.full(vapor_pressure_delta.size, 0.02897, dtype=np.float64)
    delta_p_ti, k_ti, temp_ti, molar_mass_ti = [
        ti.ndarray(ti.f64, n_particles) for _ in range(4)
    ]
    res_ti = ti.ndarray(ti.f64, n_particles)
    delta_p_ti.from_numpy(vapor_pressure_delta)
    k_ti.from_numpy(first_order_mass_transfer)
    temp_ti.from_numpy(temperature_t)
    molar_mass_ti.from_numpy(molar_mass_m)
    kget_mass_transfer_rate(delta_p_ti, k_ti, temp_ti, molar_mass_ti, res_ti)
    npt.assert_allclose(
        res_ti.to_numpy(),
        np_mr(
            vapor_pressure_delta,
            first_order_mass_transfer,
            temperature_t,
            molar_mass_m,
        ),
    )

    # ----- radius-rate kernel ---------------------------------------------
    mass_transfer_rate = np_mr(
        vapor_pressure_delta,
        first_order_mass_transfer,
        300.0,
        0.02897,
    )
    particle_density = np.full(n_particles, 1000.0, dtype=np.float64)
    dm_ti, radius_ti, density_ti = [
        ti.ndarray(ti.f64, n_particles) for _ in range(3)
    ]
    res_ti = ti.ndarray(ti.f64, n_particles)
    dm_ti.from_numpy(mass_transfer_rate)
    radius_ti.from_numpy(particle_radius)
    density_ti.from_numpy(particle_density)
    kget_radius_transfer_rate(dm_ti, radius_ti, density_ti, res_ti)
    npt.assert_allclose(
        res_ti.to_numpy(),
        np_rr(mass_transfer_rate, particle_radius, particle_density),
    )
