"""
Taichi implementation of aerodynamic mobility for particles in a fluid.

This module provides Taichi-accelerated functions and kernels to compute
the aerodynamic mobility (B) of particles, which quantifies how easily a
particle moves through a fluid under an applied force. The aerodynamic
mobility is given by:

    B = Cᶜ / (6 π μ r)

where:
    - B   : Aerodynamic mobility [m²/(V·s) or m/(N·s)]
    - Cᶜ  : Slip correction factor (dimensionless)
    - μ   : Dynamic viscosity of the fluid [Pa·s]
    - r   : Particle radius [m]

Examples:
    ```py title="Scalar Example"
    import particula as par
    B = par.backend.taichi.particles.properties.ti_aerodynamic_mobility_module.\
        ti_get_aerodynamic_mobility(1e-7, 1.2, 1.8e-5)
    # Output: 3.53e7 (example value)
    ```

    ```py title="Array Example"
    import numpy as np
    import particula as par
    radii = np.array([1e-7, 2e-7])
    slip = np.array([1.2, 1.1])
    mu = 1.8e-5
    B = par.backend.taichi.particles.properties.ti_aerodynamic_mobility_module.\
        ti_get_aerodynamic_mobility(radii, slip, mu)
    # Output: array([...])
    ```

References:
    - W.C. Hinds, "Aerosol Technology: Properties, Behavior, and Measurement
      of Airborne Particles," 2nd Edition, Wiley-Interscience, 1999.
    - "Mobility (electrical)," [Wikipedia](https://en.wikipedia.org/wiki/Mobility_(electrical)).
"""
import taichi as ti
import numpy as np
from numbers import Number
from particula.backend.dispatch_register import register

@ti.func
def fget_aerodynamic_mobility(
    particle_radius: ti.f64,
    slip_correction_factor: ti.f64,
    dynamic_viscosity: ti.f64,
) -> ti.f64:
    """
    Compute the aerodynamic mobility for a single particle (elementwise).

    The aerodynamic mobility B is calculated as:
        B = Cᶜ / (6 π μ r)
        - B   : Aerodynamic mobility [m²/(V·s) or m/(N·s)]
        - Cᶜ  : Slip correction factor (dimensionless)
        - μ   : Dynamic viscosity [Pa·s]
        - r   : Particle radius [m]

    Arguments:
        - particle_radius : Particle radius [m].
        - slip_correction_factor : Slip correction factor (dimensionless).
        - dynamic_viscosity : Dynamic viscosity of the fluid [Pa·s].

    Returns:
        - Aerodynamic mobility [m²/(V·s) or m/(N·s)].

    """
    return slip_correction_factor / (
        6.0 * ti.math.pi * dynamic_viscosity * particle_radius
    )

@ti.kernel
def kget_aerodynamic_mobility(
    particle_radius_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
    slip_correction_factor_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
    dynamic_viscosity_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Vectorized Taichi kernel for aerodynamic mobility.

    Arguments:
        - particle_radius_array : 1D ndarray of particle radii [m].
        - slip_correction_factor_array : 1D ndarray of slip correction factors.
        - dynamic_viscosity_array : 1D ndarray of dynamic viscosities [Pa·s].
        - result_array : 1D ndarray (output, in-place) for aerodynamic mobility.

    Returns:
        - None. (Results are written in-place to result_array.)
    """
    for i in range(result_array.shape[0]):
        result_array[i] = fget_aerodynamic_mobility(
            particle_radius_array[i],
            slip_correction_factor_array[i],
            dynamic_viscosity_array[i]
        )

@register("get_aerodynamic_mobility", backend="taichi")
def ti_get_aerodynamic_mobility(
    particle_radius,
    slip_correction_factor,
    dynamic_viscosity,
):
    """
    Taichi backend wrapper for aerodynamic mobility calculation.

    This function broadcasts all input arrays to a common shape, dispatches
    the computation to Taichi kernels, and returns the aerodynamic mobility
    for each particle. Supports both scalar and array inputs.

    Arguments:
        - particle_radius : Particle radius [m] (scalar or np.ndarray).
        - slip_correction_factor : Slip correction factor (scalar or np.ndarray).
        - dynamic_viscosity : Dynamic viscosity [Pa·s] (scalar or np.ndarray).

    Returns:
        - np.ndarray or float: Aerodynamic mobility, shape matches broadcasted
          input.

    Example:
        ```py
        ti_get_aerodynamic_mobility(1e-7, 1.2, 1.8e-5)
        # Output: 3.53e7 (example value)
        ```

    """
    # --- type guard -----------------------------------------------
    if not (
        isinstance(particle_radius, (np.ndarray, Number))
        and isinstance(slip_correction_factor, (np.ndarray, Number))
        and isinstance(dynamic_viscosity, (np.ndarray, Number))
    ):
        raise TypeError(
            "Taichi backend expects NumPy arrays or scalars for all inputs."
        )
    # --- broadcast -----------------------------------------------
    particle_radius_np = np.asarray(particle_radius, dtype=np.float64)
    slip_correction_factor_np = np.asarray(slip_correction_factor, dtype=np.float64)
    dynamic_viscosity_np = np.asarray(dynamic_viscosity, dtype=np.float64)
    (
        particle_radius_b,
        slip_correction_factor_b,
        dynamic_viscosity_b,
    ) = np.broadcast_arrays(
        particle_radius_np,
        slip_correction_factor_np,
        dynamic_viscosity_np,
    )

    particle_radius_flat = particle_radius_b.ravel()
    slip_correction_factor_flat = slip_correction_factor_b.ravel()
    dynamic_viscosity_flat = dynamic_viscosity_b.ravel()
    n_elements = particle_radius_flat.size

    # --- Taichi buffers ------------------------------------------
    particle_radius_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    slip_correction_factor_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    dynamic_viscosity_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    particle_radius_ti.from_numpy(particle_radius_flat)
    slip_correction_factor_ti.from_numpy(slip_correction_factor_flat)
    dynamic_viscosity_ti.from_numpy(dynamic_viscosity_flat)

    # --- kernel --------------------------------------------------
    kget_aerodynamic_mobility(
        particle_radius_ti,
        slip_correction_factor_ti,
        dynamic_viscosity_ti,
        result_ti
    )

    # --- reshape / unwrap ----------------------------------------
    result_np = result_ti.to_numpy().reshape(particle_radius_b.shape)
    return result_np.item() if result_np.size == 1 else result_np
