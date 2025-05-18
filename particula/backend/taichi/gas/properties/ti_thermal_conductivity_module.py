"""
Taichi implementation of the thermal conductivity of air.

The thermal conductivity κ [W m⁻¹ K⁻¹] is computed as:
    κ = 10⁻³ · (4.39 + 0.071 T) [W m⁻¹ K⁻¹]

Examples:
    ```py
    import numpy as np
    from particula.backend.taichi.gas.properties import (
        ti_thermal_conductivity_module as tcm
    )
    # Scalar input
    kappa = tcm.ti_get_thermal_conductivity(300.0)
    # Array input
    kappas = tcm.ti_get_thermal_conductivity(np.array([250.0, 300.0, 350.0]))
    ```

References:
    - Bergman, T.L., Lavine, A.S., Incropera, F.P., DeWitt, D.P.,
      "Fundamentals of Heat and Mass Transfer," 8th Edition, Wiley, 2020.
"""

from typing import Union
from numpy.typing import NDArray
import taichi as ti
import numpy as np

@ti.func
def fget_thermal_conductivity(temperature: ti.f64) -> ti.f64:
    """
    Compute the thermal conductivity of air for a given temperature.

    Arguments:
        - temperature : Absolute temperature [K].

    Returns:
        - κ : Thermal conductivity [W m⁻¹ K⁻¹].

    Equation:
        κ = 10⁻³ · (4.39 + 0.071 T) [W m⁻¹ K⁻¹]

    Examples:
        ```py
        kappa = fget_thermal_conductivity(300.0)
        # Output: 0.1343
        ```

    References:
        - Bergman, T.L., Lavine, A.S., Incropera, F.P., DeWitt, D.P.,
          "Fundamentals of Heat and Mass Transfer," 8th Edition, Wiley, 2020.
    """
    return 1e-3 * (4.39 + 0.071 * temperature)

@ti.kernel
def kget_thermal_conductivity(
    temperature: ti.types.ndarray(dtype=ti.f64, ndim=1),
    thermal_conductivity: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Vectorized kernel for computing thermal conductivity in-place.

    Arguments:
        - temperature : 1D array of absolute temperature [K].
        - thermal_conductivity : 1D array, output, modified in-place
          [W m⁻¹ K⁻¹].

    Returns:
        - None

    Examples:
        ```py
        import numpy as np
        temp = np.array([250.0, 300.0, 350.0])
        out = np.empty_like(temp)
        kget_thermal_conductivity(temp, out)
        # out now contains the thermal conductivities
        ```

    References:
        - Bergman, T.L., Lavine, A.S., Incropera, F.P., DeWitt, D.P.,
          "Fundamentals of Heat and Mass Transfer," 8th Edition, Wiley, 2020.
    """
    for i in range(thermal_conductivity.shape[0]):
        thermal_conductivity[i] = fget_thermal_conductivity(temperature[i])

def ti_get_thermal_conductivity(
    temperature: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]:
    """
    Taichi-accelerated wrapper for air thermal conductivity (NumPy ⇄ Taichi).

    Arguments:
        - temperature : float, list-like, or NDArray[np.float64], absolute temperature [K].

    Returns:
        - κ : float or NDArray[np.float64], thermal conductivity [W m⁻¹ K⁻¹].

    Examples:
        ```py
        kappa = ti_get_thermal_conductivity(300.0)
        # Output: 0.1343

        import numpy as np
        arr = np.array([250.0, 300.0, 350.0])
        kappas = ti_get_thermal_conductivity(arr)
        # Output: array([0.12515, 0.1343, 0.14345])
        ```

    References:
        - Bergman, T.L., Lavine, A.S., Incropera, F.P., DeWitt, D.P.,
          "Fundamentals of Heat and Mass Transfer," 8th Edition, Wiley, 2020.
    """
    # accept float, list-like or NumPy array
    temperature_array = np.atleast_1d(temperature).astype(np.float64)
    n_values = temperature_array.size

    temperature_field = ti.ndarray(dtype=ti.f64, shape=n_values)
    thermal_conductivity_field = ti.ndarray(dtype=ti.f64, shape=n_values)

    temperature_field.from_numpy(temperature_array)
    kget_thermal_conductivity(temperature_field, thermal_conductivity_field)

    thermal_conductivity_array = thermal_conductivity_field.to_numpy()
    return (
        thermal_conductivity_array.item()
        if thermal_conductivity_array.size == 1
        else thermal_conductivity_array
    )
