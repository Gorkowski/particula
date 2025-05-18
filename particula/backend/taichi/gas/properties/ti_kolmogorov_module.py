"""
Taichi implementations of Kolmogorov microscales (time, length, velocity).

Equations
---------
τₖ = √(ν ⁄ ε)          η = (ν³ ⁄ ε)¹ᐟ⁴          uₖ = √(ν × ε)

Symbols:
    ν – kinematic viscosity [m² s⁻¹]
    ε – turbulent dissipation rate [m² s⁻³]

References:
    - Pope, S. B., “Turbulent Flows”, Cambridge Univ. Press, 2000.
    - “Kolmogorov microscales,” Wikipedia.
"""

import taichi as ti
import numpy as np
from typing import Union
from numpy.typing import NDArray
from particula.backend.dispatch_register import register


@ti.func
def fget_kolmogorov_time(
    kinematic_viscosity: ti.f64, turbulent_dissipation: ti.f64
) -> ti.f64:
    """
    Kolmogorov time scale (τₖ) for turbulence.

    τₖ = √(ν ⁄ ε)

    Arguments:
        - kinematic_viscosity : ν  [m² s⁻¹]
        - turbulent_dissipation : ε [m² s⁻³]

    Returns:
        - Kolmogorov time scale τₖ [s]
    """
    return ti.sqrt(kinematic_viscosity / turbulent_dissipation)


@ti.func
def fget_kolmogorov_length(
    kinematic_viscosity: ti.f64, turbulent_dissipation: ti.f64
) -> ti.f64:
    """
    Kolmogorov length scale (η) for turbulence.

    η = (ν³ ⁄ ε)¹ᐟ⁴

    Arguments:
        - kinematic_viscosity : ν  [m² s⁻¹]
        - turbulent_dissipation : ε [m² s⁻³]

    Returns:
        - Kolmogorov length scale η [m]
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
    Kolmogorov velocity scale (uₖ) for turbulence.

    uₖ = √(ν × ε)

    Arguments:
        - kinematic_viscosity : ν  [m² s⁻¹]
        - turbulent_dissipation : ε [m² s⁻³]

    Returns:
        - Kolmogorov velocity scale uₖ [m s⁻¹]
    """
    return ti.sqrt(ti.sqrt(kinematic_viscosity * turbulent_dissipation))


@ti.kernel
def kget_kolmogorov_time(
    kinematic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    turbulent_dissipation: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
) -> None:
    """
    Vectorized Kolmogorov time kernel.

    All arrays must be 1-D and of equal length.

    Arguments:
        - kinematic_viscosity : Input array of kinematic viscosity values.
        - turbulent_dissipation : Input array of turbulent dissipation values.
        - result_array : Output array for Kolmogorov time results.
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
) -> None:
    """
    Vectorized Kolmogorov length kernel.

    All arrays must be 1-D and of equal length.

    Arguments:
        - kinematic_viscosity : Input array of kinematic viscosity values.
        - turbulent_dissipation : Input array of turbulent dissipation values.
        - result_array : Output array for Kolmogorov length results.
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
) -> None:
    """
    Vectorized Kolmogorov velocity kernel.

    All arrays must be 1-D and of equal length.

    Arguments:
        - kinematic_viscosity : Input array of kinematic viscosity values.
        - turbulent_dissipation : Input array of turbulent dissipation values.
        - result_array : Output array for Kolmogorov velocity results.
    """
    for index in range(result_array.shape[0]):
        result_array[index] = fget_kolmogorov_velocity(
            kinematic_viscosity[index], turbulent_dissipation[index]
        )


@register("get_kolmogorov_time", backend="taichi")
def ti_get_kolmogorov_time(
    kinematic_viscosity: Union[float, NDArray[np.float64]],
    turbulent_dissipation: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Taichi wrapper for Kolmogorov time scale.

    Both inputs must be equal-length 1-D NumPy arrays or scalars; no
    internal broadcasting is performed. If the result has size 1, a scalar
    is returned.

    Arguments:
        - kinematic_viscosity : NumPy 1-D array or float (ν)
        - turbulent_dissipation : NumPy 1-D array or float (ε)

    Returns:
        - Kolmogorov time scale τₖ as a NumPy array, or a scalar if result
          size is 1.

    Examples:
        ```py
        import numpy as np
        ti_get_kolmogorov_time(np.array([1.5e-5]), np.array([1e-3]))
        # Output: array([0.12247449])
        ```

    References:
        - Pope, S. B., "Turbulent Flows," Cambridge University Press, 2000.
        - "Kolmogorov microscales,"
          https://en.wikipedia.org/wiki/Kolmogorov_microscales
    """
    if not (
        isinstance(kinematic_viscosity, np.ndarray)
        and isinstance(turbulent_dissipation, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")
    kinematic_viscosity_array = np.atleast_1d(kinematic_viscosity)
    turbulent_dissipation_array = np.atleast_1d(turbulent_dissipation)
    n_elements = kinematic_viscosity_array.size
    kinematic_viscosity_field = ti.ndarray(dtype=ti.f64, shape=n_elements)
    turbulent_dissipation_field = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_field = ti.ndarray(dtype=ti.f64, shape=n_elements)
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
def ti_get_kolmogorov_length(
    kinematic_viscosity: Union[float, NDArray[np.float64]],
    turbulent_dissipation: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Taichi wrapper for Kolmogorov length scale.

    Both inputs must be equal-length 1-D NumPy arrays or scalars; no
    internal broadcasting is performed. If the result has size 1, a scalar
    is returned.

    Arguments:
        - kinematic_viscosity : NumPy 1-D array or float (ν)
        - turbulent_dissipation : NumPy 1-D array or float (ε)

    Returns:
        - Kolmogorov length scale η as a NumPy array, or a scalar if result
          size is 1.

    Examples:
        ```py
        import numpy as np
        ti_get_kolmogorov_length(np.array([1.5e-5]), np.array([1e-3]))
        # Output: array([0.06062178])
        ```

    References:
        - Pope, S. B., "Turbulent Flows," Cambridge University Press, 2000.
        - "Kolmogorov microscales,"
          https://en.wikipedia.org/wiki/Kolmogorov_microscales
    """
    if not (
        isinstance(kinematic_viscosity, np.ndarray)
        and isinstance(turbulent_dissipation, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")
    kinematic_viscosity_array = np.atleast_1d(kinematic_viscosity)
    turbulent_dissipation_array = np.atleast_1d(turbulent_dissipation)
    n_elements = kinematic_viscosity_array.size
    kinematic_viscosity_field = ti.ndarray(dtype=ti.f64, shape=n_elements)
    turbulent_dissipation_field = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_field = ti.ndarray(dtype=ti.f64, shape=n_elements)
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
def ti_get_kolmogorov_velocity(
    kinematic_viscosity: Union[float, NDArray[np.float64]],
    turbulent_dissipation: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Taichi wrapper for Kolmogorov velocity scale.

    Both inputs must be equal-length 1-D NumPy arrays or scalars; no
    internal broadcasting is performed. If the result has size 1, a scalar
    is returned.

    Arguments:
        - kinematic_viscosity : NumPy 1-D array or float (ν)
        - turbulent_dissipation : NumPy 1-D array or float (ε)

    Returns:
        - Kolmogorov velocity scale uₖ as a NumPy array, or a scalar if
          result size is 1.

    Examples:
        ```py
        import numpy as np
        ti_get_kolmogorov_velocity(np.array([1.5e-5]), np.array([1e-3]))
        # Output: array([0.07745967])
        ```

    References:
        - Pope, S. B., "Turbulent Flows," Cambridge University Press, 2000.
        - "Kolmogorov microscales,"
          https://en.wikipedia.org/wiki/Kolmogorov_microscales
    """
    if not (
        isinstance(kinematic_viscosity, np.ndarray)
        and isinstance(turbulent_dissipation, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")
    kinematic_viscosity_array = np.atleast_1d(kinematic_viscosity)
    turbulent_dissipation_array = np.atleast_1d(turbulent_dissipation)
    n_elements = kinematic_viscosity_array.size
    kinematic_viscosity_field = ti.ndarray(dtype=ti.f64, shape=n_elements)
    turbulent_dissipation_field = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_field = ti.ndarray(dtype=ti.f64, shape=n_elements)
    kinematic_viscosity_field.from_numpy(kinematic_viscosity_array)
    turbulent_dissipation_field.from_numpy(turbulent_dissipation_array)
    kget_kolmogorov_velocity(
        kinematic_viscosity_field,
        turbulent_dissipation_field,
        result_field,
    )
    result_array = result_field.to_numpy()
    return result_array.item() if result_array.size == 1 else result_array
