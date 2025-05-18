"""Taichi-accelerated Kolmogorov scales for gas properties."""

import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register


@ti.func
def fget_kolmogorov_time(
    kinematic_viscosity: ti.f64, turbulent_dissipation: ti.f64
) -> ti.f64:
    """
    Compute the Kolmogorov time scale for turbulence.

    The Kolmogorov time scale τₖ is given by:
        τₖ = √(ν∕ε)
    where:
        - τₖ is the Kolmogorov time scale,
        - ν is the kinematic viscosity,
        - ε is the turbulent dissipation rate.

    Arguments:
        - kinematic_viscosity : Kinematic viscosity (ν) [m²/s].
        - turbulent_dissipation : Turbulent dissipation rate (ε) [m²/s³].

    Returns:
        - Kolmogorov time scale τₖ [s].

    Examples:
        ```py
        fget_kolmogorov_time(1.5e-5, 1e-3)
        # Output: 0.12247448713915891
        ```

    References:
        - Pope, S. B., "Turbulent Flows," Cambridge University Press, 2000.
        - "Kolmogorov microscales," [Wikipedia](https://en.wikipedia.org/wiki/Kolmogorov_microscales).
    """
    return ti.sqrt(kinematic_viscosity / turbulent_dissipation)


@ti.func
def fget_kolmogorov_length(
    kinematic_viscosity: ti.f64, turbulent_dissipation: ti.f64
) -> ti.f64:
    """
    Compute the Kolmogorov length scale for turbulence.

    The Kolmogorov length scale η is given by:
        η = (ν³∕ε)¼
    where:
        - η is the Kolmogorov length scale,
        - ν is the kinematic viscosity,
        - ε is the turbulent dissipation rate.

    Arguments:
        - kinematic_viscosity : Kinematic viscosity (ν) [m²/s].
        - turbulent_dissipation : Turbulent dissipation rate (ε) [m²/s³].

    Returns:
        - Kolmogorov length scale η [m].

    Examples:
        ```py
        fget_kolmogorov_length(1.5e-5, 1e-3)
        # Output: 0.0606217782649107
        ```

    References:
        - Pope, S. B., "Turbulent Flows," Cambridge University Press, 2000.
        - "Kolmogorov microscales," [Wikipedia](https://en.wikipedia.org/wiki/Kolmogorov_microscales).
    """
    return ti.sqrt(
        ti.sqrt(
            kinematic_viscosity
            * kinematic_viscosity
            * kinematic_viscosity
            / turbulent_dissipation
        )
    )


@ti.func
def fget_kolmogorov_velocity(
    kinematic_viscosity: ti.f64, turbulent_dissipation: ti.f64
) -> ti.f64:
    """
    Compute the Kolmogorov velocity scale for turbulence.

    The Kolmogorov velocity scale uₖ is given by:
        uₖ = (ν ε)¼
    where:
        - uₖ is the Kolmogorov velocity scale,
        - ν is the kinematic viscosity,
        - ε is the turbulent dissipation rate.

    Arguments:
        - kinematic_viscosity : Kinematic viscosity (ν) [m²/s].
        - turbulent_dissipation : Turbulent dissipation rate (ε) [m²/s³].

    Returns:
        - Kolmogorov velocity scale uₖ [m/s].

    Examples:
        ```py
        fget_kolmogorov_velocity(1.5e-5, 1e-3)
        # Output: 0.07745966692414834
        ```

    References:
        - Pope, S. B., "Turbulent Flows," Cambridge University Press, 2000.
        - "Kolmogorov microscales," [Wikipedia](https://en.wikipedia.org/wiki/Kolmogorov_microscales).
    """
    return ti.sqrt(ti.sqrt(kinematic_viscosity * turbulent_dissipation))


@ti.kernel
def kget_kolmogorov_time(
    kinematic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    turbulent_dissipation: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Vectorized Kolmogorov time kernel.

    Arguments:
        - kinematic_viscosity : Input array of kinematic viscosity values.
        - turbulent_dissipation : Input array of turbulent dissipation values.
        - result_array : Output array for Kolmogorov time results.

    Returns:
        - None. Results are written to result_array.
    """
    for index in range(result_array.shape[0]):
        result_array[index] = fget_kolmogorov_time(
            kinematic_viscosity[index], turbulent_dissipation[index]
        )


@ti.kernel
def kget_kolmogorov_length(
    kinematic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    turbulent_dissipation: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Vectorized Kolmogorov length kernel.

    Arguments:
        - kinematic_viscosity : Input array of kinematic viscosity values.
        - turbulent_dissipation : Input array of turbulent dissipation values.
        - result_array : Output array for Kolmogorov length results.

    Returns:
        - None. Results are written to result_array.
    """
    for index in range(result_array.shape[0]):
        result_array[index] = fget_kolmogorov_length(
            kinematic_viscosity[index], turbulent_dissipation[index]
        )


@ti.kernel
def kget_kolmogorov_velocity(
    kinematic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    turbulent_dissipation: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Vectorized Kolmogorov velocity kernel.

    Arguments:
        - kinematic_viscosity : Input array of kinematic viscosity values.
        - turbulent_dissipation : Input array of turbulent dissipation values.
        - result_array : Output array for Kolmogorov velocity results.

    Returns:
        - None. Results are written to result_array.
    """
    for index in range(result_array.shape[0]):
        result_array[index] = fget_kolmogorov_velocity(
            kinematic_viscosity[index], turbulent_dissipation[index]
        )


@register("get_kolmogorov_time", backend="taichi")
def ti_get_kolmogorov_time(kinematic_viscosity, turbulent_dissipation):
    """
    Taichi wrapper for Kolmogorov time scale.

    Both input arrays must be NumPy 1-D arrays. Shapes are broadcast internally.
    If the result has size 1, a scalar is returned.

    Arguments:
        - kinematic_viscosity : NumPy 1-D array of kinematic viscosity values (ν)
        - turbulent_dissipation : NumPy 1-D array of turbulent dissipation
          values (ε)

    Returns:
        - Kolmogorov time scale τₖ as a NumPy array, or a scalar if result size
          is 1.

    Examples:
        ```py
        import numpy as np
        ti_get_kolmogorov_time(np.array([1.5e-5]), np.array([1e-3]))
        # Output: array([0.12247449])
        ```

    References:
        - Pope, S. B., "Turbulent Flows," Cambridge University Press, 2000.
        - "Kolmogorov microscales," [Wikipedia](https://en.wikipedia.org/wiki/Kolmogorov_microscales).
    """
    if not (
        isinstance(kinematic_viscosity, np.ndarray)
        and isinstance(turbulent_dissipation, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")
    kinematic_viscosity_array = np.atleast_1d(kinematic_viscosity)
    turbulent_dissipation_array = np.atleast_1d(turbulent_dissipation)
    n_values = kinematic_viscosity_array.size
    kinematic_viscosity_field = ti.ndarray(dtype=ti.f64, shape=n_values)
    turbulent_dissipation_field = ti.ndarray(dtype=ti.f64, shape=n_values)
    result_field = ti.ndarray(dtype=ti.f64, shape=n_values)
    kinematic_viscosity_field.from_numpy(kinematic_viscosity_array)
    turbulent_dissipation_field.from_numpy(turbulent_dissipation_array)
    kget_kolmogorov_time(
        kinematic_viscosity_field,
        turbulent_dissipation_field,
        result_field,
    )
    result_array = result_field.to_numpy()
    return result_array.item() if result_array.size == 1 else result_array


@register("get_kolmogorov_length", backend="taichi")
def ti_get_kolmogorov_length(kinematic_viscosity, turbulent_dissipation):
    """
    Taichi wrapper for Kolmogorov length scale.

    Both input arrays must be NumPy 1-D arrays. Shapes are broadcast internally.
    If the result has size 1, a scalar is returned.

    Arguments:
        - kinematic_viscosity : NumPy 1-D array of kinematic viscosity values (ν)
        - turbulent_dissipation : NumPy 1-D array of turbulent dissipation
          values (ε)

    Returns:
        - Kolmogorov length scale η as a NumPy array, or a scalar if result size
          is 1.

    Examples:
        ```py
        import numpy as np
        ti_get_kolmogorov_length(np.array([1.5e-5]), np.array([1e-3]))
        # Output: array([0.06062178])
        ```

    References:
        - Pope, S. B., "Turbulent Flows," Cambridge University Press, 2000.
        - "Kolmogorov microscales," [Wikipedia](https://en.wikipedia.org/wiki/Kolmogorov_microscales).
    """
    if not (
        isinstance(kinematic_viscosity, np.ndarray)
        and isinstance(turbulent_dissipation, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")
    kinematic_viscosity_array = np.atleast_1d(kinematic_viscosity)
    turbulent_dissipation_array = np.atleast_1d(turbulent_dissipation)
    n_values = kinematic_viscosity_array.size
    kinematic_viscosity_field = ti.ndarray(dtype=ti.f64, shape=n_values)
    turbulent_dissipation_field = ti.ndarray(dtype=ti.f64, shape=n_values)
    result_field = ti.ndarray(dtype=ti.f64, shape=n_values)
    kinematic_viscosity_field.from_numpy(kinematic_viscosity_array)
    turbulent_dissipation_field.from_numpy(turbulent_dissipation_array)
    kget_kolmogorov_length(
        kinematic_viscosity_field,
        turbulent_dissipation_field,
        result_field,
    )
    result_array = result_field.to_numpy()
    return result_array.item() if result_array.size == 1 else result_array


@register("get_kolmogorov_velocity", backend="taichi")
def ti_get_kolmogorov_velocity(kinematic_viscosity, turbulent_dissipation):
    """
    Taichi wrapper for Kolmogorov velocity scale.

    Both input arrays must be NumPy 1-D arrays. Shapes are broadcast internally.
    If the result has size 1, a scalar is returned.

    Arguments:
        - kinematic_viscosity : NumPy 1-D array of kinematic viscosity values (ν)
        - turbulent_dissipation : NumPy 1-D array of turbulent dissipation
          values (ε)

    Returns:
        - Kolmogorov velocity scale uₖ as a NumPy array, or a scalar if result
          size is 1.

    Examples:
        ```py
        import numpy as np
        ti_get_kolmogorov_velocity(np.array([1.5e-5]), np.array([1e-3]))
        # Output: array([0.07745967])
        ```

    References:
        - Pope, S. B., "Turbulent Flows," Cambridge University Press, 2000.
        - "Kolmogorov microscales," [Wikipedia](https://en.wikipedia.org/wiki/Kolmogorov_microscales).
    """
    if not (
        isinstance(kinematic_viscosity, np.ndarray)
        and isinstance(turbulent_dissipation, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")
    kinematic_viscosity_array = np.atleast_1d(kinematic_viscosity)
    turbulent_dissipation_array = np.atleast_1d(turbulent_dissipation)
    n_values = kinematic_viscosity_array.size
    kinematic_viscosity_field = ti.ndarray(dtype=ti.f64, shape=n_values)
    turbulent_dissipation_field = ti.ndarray(dtype=ti.f64, shape=n_values)
    result_field = ti.ndarray(dtype=ti.f64, shape=n_values)
    kinematic_viscosity_field.from_numpy(kinematic_viscosity_array)
    turbulent_dissipation_field.from_numpy(turbulent_dissipation_array)
    kget_kolmogorov_velocity(
        kinematic_viscosity_field,
        turbulent_dissipation_field,
        result_field,
    )
    result_array = result_field.to_numpy()
    return result_array.item() if result_array.size == 1 else result_array
