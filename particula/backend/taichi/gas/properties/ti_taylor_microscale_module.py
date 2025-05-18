"""Taichi-accelerated Taylor microscale helpers."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

@ti.func
def fget_lagrangian_taylor_microscale_time(
    kolmogorov_time: ti.f64,
    taylor_microscale_reynolds_number: ti.f64,
    acceleration_variance: ti.f64
) -> ti.f64:
    """
    Compute the Lagrangian Taylor microscale time (elementwise, Taichi).

    Calculates the characteristic time scale for turbulent flows using
    the Kolmogorov time, Taylor microscale Reynolds number, and
    acceleration variance.

    Arguments:
        - kolmogorov_time : Kolmogorov time scale, in seconds.
        - taylor_microscale_reynolds_number : Taylor microscale Reynolds
          number, dimensionless.
        - acceleration_variance : Acceleration variance, in (m/s²)².

    Returns:
        - lagrangian_taylor_microscale_time : Lagrangian Taylor microscale
          time, in seconds.

    Examples:
        ```py title="Quick example"
        import numpy as np
        τ_λ = get_lagrangian_taylor_microscale_time(
            np.array([0.01]), np.array([120]), np.array([1.5])
        )
        ```
    References:
        - G. I. Taylor, "Statistical theory of turbulence," Proc. Roy. Soc. A,
          151, 421–444, 1935.
    """
    return kolmogorov_time * ti.sqrt(
        (2.0 * taylor_microscale_reynolds_number) /
        (ti.sqrt(15.0) * acceleration_variance)
    )

@ti.func
def fget_taylor_microscale(
    fluid_rms_velocity: ti.f64,
    kinematic_viscosity: ti.f64,
    turbulent_dissipation: ti.f64
) -> ti.f64:
    """
    Compute the Taylor microscale (elementwise, Taichi).

    Calculates the Taylor microscale for turbulent flows using the
    fluid RMS velocity, kinematic viscosity, and turbulent dissipation.

    Arguments:
        - fluid_rms_velocity : Root-mean-square fluid velocity, in m/s.
        - kinematic_viscosity : Kinematic viscosity, in m²/s.
        - turbulent_dissipation : Turbulent dissipation rate, in m²/s³.

    Returns:
        - taylor_microscale : Taylor microscale, in meters.

    Examples:
        ```py title="Quick example"
        import numpy as np
        λ = get_taylor_microscale(
            np.array([0.5]), np.array([1e-6]), np.array([1e-3])
        )
        ```
    References:
        - G. I. Taylor, "Statistical theory of turbulence," Proc. Roy. Soc. A,
          151, 421–444, 1935.
    """
    return fluid_rms_velocity * ti.sqrt(
        (15.0 * kinematic_viscosity**2) / turbulent_dissipation
    )

@ti.func
def fget_taylor_microscale_reynolds_number(
    fluid_rms_velocity: ti.f64,
    taylor_microscale: ti.f64,
    kinematic_viscosity: ti.f64
) -> ti.f64:
    """
    Compute the Taylor-microscale Reynolds number (elementwise, Taichi).

    Calculates the Reynolds number based on the Taylor microscale,
    fluid RMS velocity, and kinematic viscosity.

    Arguments:
        - fluid_rms_velocity : Root-mean-square fluid velocity, in m/s.
        - taylor_microscale : Taylor microscale, in meters.
        - kinematic_viscosity : Kinematic viscosity, in m²/s.

    Returns:
        - taylor_microscale_reynolds_number : Taylor-microscale Reynolds
          number, dimensionless.

    Examples:
        ```py title="Quick example"
        import numpy as np
        Re_λ = get_taylor_microscale_reynolds_number(
            np.array([0.5]), np.array([0.01]), np.array([1e-6])
        )
        ```
    References:
        - G. I. Taylor, "Statistical theory of turbulence," Proc. Roy. Soc. A,
          151, 421–444, 1935.
    """
    return (fluid_rms_velocity * taylor_microscale) / kinematic_viscosity

@ti.kernel
def kget_lagrangian_taylor_microscale_time(
    kolmogorov_time_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
    taylor_microscale_reynolds_number_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
    acceleration_variance_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Vectorized Lagrangian Taylor microscale time (Taichi).

    Computes the Lagrangian Taylor microscale time for each element in
    the input arrays.

    Arguments:
        - kolmogorov_time_array : 1D array of Kolmogorov time scales, s.
        - taylor_microscale_reynolds_number_array : 1D array of Taylor
          microscale Reynolds numbers, dimensionless.
        - acceleration_variance_array : 1D array of acceleration
          variances, (m/s²)².
        - result_array : Output array for Lagrangian Taylor microscale
          times, s.

    Returns:
        - None (results stored in result_array)

    Examples:
        ```py title="Quick example"
        import numpy as np
        # See ti_get_lagrangian_taylor_microscale_time for usage.
        ```
    References:
        - G. I. Taylor, "Statistical theory of turbulence," Proc. Roy. Soc. A,
          151, 421–444, 1935.
    """
    for i in range(result_array.shape[0]):
        result_array[i] = fget_lagrangian_taylor_microscale_time(
            kolmogorov_time_array[i],
            taylor_microscale_reynolds_number_array[i],
            acceleration_variance_array[i]
        )

@ti.kernel
def kget_taylor_microscale(
    fluid_rms_velocity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    kinematic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    turbulent_dissipation: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Vectorized Taylor microscale (Taichi).

    Computes the Taylor microscale for each element in the input arrays.

    Arguments:
        - fluid_rms_velocity : 1D array of RMS fluid velocities, m/s.
        - kinematic_viscosity : 1D array of kinematic viscosities, m²/s.
        - turbulent_dissipation : 1D array of turbulent dissipation rates,
          m²/s³.
        - result : Output array for Taylor microscales, m.

    Returns:
        - None (results stored in result)

    Examples:
        ```py title="Quick example"
        import numpy as np
        # See ti_get_taylor_microscale for usage.
        ```
    References:
        - G. I. Taylor, "Statistical theory of turbulence," Proc. Roy. Soc. A,
          151, 421–444, 1935.
    """
    for i in range(result.shape[0]):
        result[i] = fget_taylor_microscale(
            fluid_rms_velocity[i],
            kinematic_viscosity[i],
            turbulent_dissipation[i],
        )

@ti.kernel
def kget_taylor_microscale_reynolds_number(
    fluid_rms_velocity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    taylor_microscale: ti.types.ndarray(dtype=ti.f64, ndim=1),
    kinematic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Vectorized Taylor-microscale Reynolds number (Taichi).

    Computes the Taylor-microscale Reynolds number for each element in
    the input arrays.

    Arguments:
        - fluid_rms_velocity : 1D array of RMS fluid velocities, m/s.
        - taylor_microscale : 1D array of Taylor microscales, m.
        - kinematic_viscosity : 1D array of kinematic viscosities, m²/s.
        - result : Output array for Taylor-microscale Reynolds numbers,
          dimensionless.

    Returns:
        - None (results stored in result)

    Examples:
        ```py title="Quick example"
        import numpy as np
        # See ti_get_taylor_microscale_reynolds_number for usage.
        ```
    References:
        - G. I. Taylor, "Statistical theory of turbulence," Proc. Roy. Soc. A,
          151, 421–444, 1935.
    """
    for i in range(result.shape[0]):
        result[i] = fget_taylor_microscale_reynolds_number(
            fluid_rms_velocity[i],
            taylor_microscale[i],
            kinematic_viscosity[i],
        )

@register("get_lagrangian_taylor_microscale_time", backend="taichi")
def ti_get_lagrangian_taylor_microscale_time(
    kolmogorov_time, taylor_microscale_reynolds_number, acceleration_variance
):
    """
    Taichi wrapper for Lagrangian Taylor microscale time.

    Accepts scalar or array-like input for all arguments. Non-array
    inputs are internally converted to 1D float64 NumPy arrays.

    Arguments:
        - kolmogorov_time : Kolmogorov time scale, s (scalar or array).
        - taylor_microscale_reynolds_number : Taylor microscale Reynolds
          number, dimensionless (scalar or array).
        - acceleration_variance : Acceleration variance, (m/s²)²
          (scalar or array).

    Returns:
        - lagrangian_taylor_microscale_time : Lagrangian Taylor
          microscale time, s (NumPy array or scalar).

    Examples:
        ```py title="Quick example"
        import numpy as np
        τ_λ = get_lagrangian_taylor_microscale_time(
            0.01, 120, 1.5
        )
        # or with arrays:
        τ_λ = get_lagrangian_taylor_microscale_time(
            np.array([0.01]), np.array([120]), np.array([1.5])
        )
        ```
    References:
        - G. I. Taylor, "Statistical theory of turbulence," Proc. Roy. Soc. A,
          151, 421–444, 1935.
    """
    # Accept scalar or array-like, then force 1-D float64 NumPy arrays
    kolmogorov_time_array = np.atleast_1d(
        np.asarray(kolmogorov_time, dtype=float)
    )
    taylor_microscale_reynolds_number_array = np.atleast_1d(
        np.asarray(taylor_microscale_reynolds_number, dtype=float)
    )
    acceleration_variance_array = np.atleast_1d(
        np.asarray(acceleration_variance, dtype=float)
    )
    n_values = kolmogorov_time_array.size

    kolmogorov_time_ti = ti.ndarray(dtype=ti.f64, shape=n_values)
    taylor_microscale_reynolds_number_ti = ti.ndarray(dtype=ti.f64, shape=n_values)
    acceleration_variance_ti = ti.ndarray(dtype=ti.f64, shape=n_values)
    result_ti_array = ti.ndarray(dtype=ti.f64, shape=n_values)
    kolmogorov_time_ti.from_numpy(kolmogorov_time_array)
    taylor_microscale_reynolds_number_ti.from_numpy(
        taylor_microscale_reynolds_number_array
    )
    acceleration_variance_ti.from_numpy(acceleration_variance_array)

    kget_lagrangian_taylor_microscale_time(
        kolmogorov_time_ti,
        taylor_microscale_reynolds_number_ti,
        acceleration_variance_ti,
        result_ti_array,
    )
    result_np = result_ti_array.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np

@register("get_taylor_microscale", backend="taichi")
def ti_get_taylor_microscale(
    fluid_rms_velocity, kinematic_viscosity, turbulent_dissipation
):
    """
    Taichi wrapper for Taylor microscale.

    Accepts scalar or array-like input for all arguments. Non-array
    inputs are internally converted to 1D float64 NumPy arrays.

    Arguments:
        - fluid_rms_velocity : Root-mean-square fluid velocity, m/s
          (scalar or array).
        - kinematic_viscosity : Kinematic viscosity, m²/s (scalar or array).
        - turbulent_dissipation : Turbulent dissipation rate, m²/s³
          (scalar or array).

    Returns:
        - taylor_microscale : Taylor microscale, m (NumPy array or scalar).

    Examples:
        ```py title="Quick example"
        import numpy as np
        λ = get_taylor_microscale(
            0.5, 1e-6, 1e-3
        )
        # or with arrays:
        λ = get_taylor_microscale(
            np.array([0.5]), np.array([1e-6]), np.array([1e-3])
        )
        ```
    References:
        - G. I. Taylor, "Statistical theory of turbulence," Proc. Roy. Soc. A,
          151, 421–444, 1935.
    """
    # Accept scalar or array-like, then force 1-D float64 NumPy arrays
    fluid_rms_velocity_array = np.atleast_1d(
        np.asarray(fluid_rms_velocity, dtype=float)
    )
    kinematic_viscosity_array = np.atleast_1d(
        np.asarray(kinematic_viscosity, dtype=float)
    )
    turbulent_dissipation_array = np.atleast_1d(
        np.asarray(turbulent_dissipation, dtype=float)
    )
    n_values = fluid_rms_velocity_array.size

    fluid_rms_velocity_ti = ti.ndarray(dtype=ti.f64, shape=n_values)
    kinematic_viscosity_ti = ti.ndarray(dtype=ti.f64, shape=n_values)
    turbulent_dissipation_ti = ti.ndarray(dtype=ti.f64, shape=n_values)
    result_ti_array = ti.ndarray(dtype=ti.f64, shape=n_values)
    fluid_rms_velocity_ti.from_numpy(fluid_rms_velocity_array)
    kinematic_viscosity_ti.from_numpy(kinematic_viscosity_array)
    turbulent_dissipation_ti.from_numpy(turbulent_dissipation_array)

    kget_taylor_microscale(
        fluid_rms_velocity_ti,
        kinematic_viscosity_ti,
        turbulent_dissipation_ti,
        result_ti_array,
    )
    result_np = result_ti_array.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np

@register("get_taylor_microscale_reynolds_number", backend="taichi")
def ti_get_taylor_microscale_reynolds_number(
    fluid_rms_velocity, taylor_microscale, kinematic_viscosity
):
    """
    Taichi wrapper for Taylor-microscale Reynolds number.

    Accepts scalar or array-like input for all arguments. Non-array
    inputs are internally converted to 1D float64 NumPy arrays.

    Arguments:
        - fluid_rms_velocity : Root-mean-square fluid velocity, m/s
          (scalar or array).
        - taylor_microscale : Taylor microscale, m (scalar or array).
        - kinematic_viscosity : Kinematic viscosity, m²/s (scalar or array).

    Returns:
        - taylor_microscale_reynolds_number : Taylor-microscale Reynolds
          number, dimensionless (NumPy array or scalar).

    Examples:
        ```py title="Quick example"
        import numpy as np
        Re_λ = get_taylor_microscale_reynolds_number(
            0.5, 0.01, 1e-6
        )
        # or with arrays:
        Re_λ = get_taylor_microscale_reynolds_number(
            np.array([0.5]), np.array([0.01]), np.array([1e-6])
        )
        ```
    References:
        - G. I. Taylor, "Statistical theory of turbulence," Proc. Roy. Soc. A,
          151, 421–444, 1935.
    """
    # Accept scalar or array-like, then force 1-D float64 NumPy arrays
    fluid_rms_velocity_array = np.atleast_1d(
        np.asarray(fluid_rms_velocity, dtype=float)
    )
    taylor_microscale_array = np.atleast_1d(
        np.asarray(taylor_microscale, dtype=float)
    )
    kinematic_viscosity_array = np.atleast_1d(
        np.asarray(kinematic_viscosity, dtype=float)
    )
    n_values = fluid_rms_velocity_array.size

    fluid_rms_velocity_ti = ti.ndarray(dtype=ti.f64, shape=n_values)
    taylor_microscale_ti = ti.ndarray(dtype=ti.f64, shape=n_values)
    kinematic_viscosity_ti = ti.ndarray(dtype=ti.f64, shape=n_values)
    result_ti_array = ti.ndarray(dtype=ti.f64, shape=n_values)
    fluid_rms_velocity_ti.from_numpy(fluid_rms_velocity_array)
    taylor_microscale_ti.from_numpy(taylor_microscale_array)
    kinematic_viscosity_ti.from_numpy(kinematic_viscosity_array)

    kget_taylor_microscale_reynolds_number(
        fluid_rms_velocity_ti,
        taylor_microscale_ti,
        kinematic_viscosity_ti,
        result_ti_array,
    )
    result_np = result_ti_array.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np
