"""
Vapor pressure modules for calculating the vapor pressure of a gas.
"""
from typing import Union
import numpy as np
from numpy.typing import NDArray

MMHG_TO_PA = 133.32238741499998  # mmHg → Pa conversion factor

from particula.util.constants import GAS_CONSTANT
from particula.util.validate_inputs import validate_inputs


@validate_inputs(
    {
        "a": "finite",
        "b": "finite",
        "c": "finite",
        "temperature": "positive",
    }
)
def get_antoine_vapor_pressure(
    a: Union[float, NDArray[np.float64]],
    b: Union[float, NDArray[np.float64]],
    c: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate vapor pressure using the Antoine equation.

    The Antoine equation relates the logarithm of vapor pressure to
    temperature for a pure substance.

    - P = 10^(constant_a - constant_b / (T - constant_c)) × MMHG_TO_PA
        - P is Vapor pressure [Pa].
        - constant_a, constant_b, constant_c are Antoine equation parameters (dimensionless).
        - T is Temperature [K].

    Arguments:
        - constant_a : Antoine parameter A (dimensionless).
        - constant_b : Antoine parameter B (dimensionless).
        - constant_c : Antoine parameter C (dimensionless).
        - temperature : Temperature in Kelvin [K].

    Returns:
        - Vapor pressure in Pascals [Pa].

    Examples:
        ```py title="Example usage"
        import particula as par
        par.gas.get_antoine_vapor_pressure(
            8.07131, 1730.63, 233.426, 373.15
        )
        # Output: ~101325 Pa (roughly 1 atm)
        ```

    References:
        - https://en.wikipedia.org/wiki/Antoine_equation
        - Kelvin conversion details:
          https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118135341.app1
    """
    vapor_pressure_log = a - (b / (temperature - c))
    vapor_pressure = 10 ** vapor_pressure_log
    return vapor_pressure * MMHG_TO_PA


@validate_inputs(
    {
        "latent_heat": "positive",
        "temperature_initial": "positive",
        "pressure_initial": "nonnegative",
        "temperature": "positive",
    }
)
def get_clausius_clapeyron_vapor_pressure(
    latent_heat: Union[float, NDArray[np.float64]],
    temperature_initial: Union[float, NDArray[np.float64]],
    pressure_initial: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
    gas_constant: float = GAS_CONSTANT,
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate vapor pressure using Clausius-Clapeyron equation.

    This function calculates the final vapor pressure based on an initial
    temperature/pressure pair and the latent heat of vaporization,
    assuming ideal gas behavior.

    - P_final = P_initial × exp( (L / R) × (1 / T_initial - 1 / T_final) )
        - P_final is Final vapor pressure [Pa].
        - P_initial is Initial vapor pressure [Pa].
        - L is Latent heat of vaporization [J/mol].
        - R is Universal gas constant [J/(mol·K)].
        - T_initial is Initial temperature [K].
        - T_final is Final temperature [K].

    Arguments:
        - latent_heat : Latent heat of vaporization [J/mol].
        - temperature_initial : Initial temperature [K].
        - pressure_initial : Initial vapor pressure [Pa].
        - temperature : Final temperature [K].
        - gas_constant : Gas constant (default 8.314 J/(mol·K)).

    Returns:
        - Pure vapor pressure [Pa].

    Examples:
        ```py title="Example usage"
        import particula as par
        par.gas.get_clausius_clapeyron_vapor_pressure(
            40660, 373.15, 101325, 300
        )
        # Output: ~35307 Pa
        ```

    References:
        - https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation
    """
    return pressure_initial * np.exp(
        (latent_heat / gas_constant)
        * (1 / temperature_initial - 1 / temperature)
    )


@validate_inputs(
    {
        "temperature": "positive",
    }
)
def get_buck_vapor_pressure(
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate vapor pressure using the Buck equation for water vapor.

    Uses separate empirical formulas below 0 °C and above 0 °C to compute
    water vapor pressure.

    - For temperature_celsius < 0 °C:
        vapor_pressure = 6.1115 × exp( (23.036 - temperature_celsius/333.7) × temperature_celsius / (279.82 + temperature_celsius) ) × 100
    - For temperature_celsius ≥ 0 °C:
        vapor_pressure = 6.1121 × exp( (18.678 - temperature_celsius/234.5) × temperature_celsius / (257.14 + temperature_celsius) ) × 100
        - vapor_pressure is Vapor pressure [Pa].
        - temperature_celsius is Temperature in Celsius [°C] (converted internally from Kelvin).

    Arguments:
        - temperature : Temperature in Kelvin [K].

    Returns:
        - Vapor pressure in Pascals [Pa].

    Examples:
        ```py title="Example usage"
        import particula as par
        par.gas.get_buck_vapor_pressure(273.15)
        # Output: ~611 Pa (around ice point)
        ```

    References:
        - Buck, A. L., (1981)
        - https://en.wikipedia.org/wiki/Arden_Buck_equation
    """
    temperature_celsius = np.asarray(temperature) - 273.15
    mask_below_freezing = temperature_celsius < 0.0
    mask_above_freezing = ~mask_below_freezing

    vapor_pressure_below_freezing = (
        6.1115 * np.exp(
            (23.036 - temperature_celsius / 333.7)
            * temperature_celsius
            / (279.82 + temperature_celsius)
        ) * 100
    )
    vapor_pressure_above_freezing = (
        6.1121 * np.exp(
            (18.678 - temperature_celsius / 234.5)
            * temperature_celsius
            / (257.14 + temperature_celsius)
        ) * 100
    )

    return (
        vapor_pressure_below_freezing * mask_below_freezing
        + vapor_pressure_above_freezing * mask_above_freezing
    )
