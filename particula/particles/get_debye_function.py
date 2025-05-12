"""
Debye function implementation for use in wall loss coefficient calculations.

This module provides a function to compute the Debye function, which is used
in the calculation of wall loss coefficients for particles in chambers.
"""

import numpy as np
from typing import Union
from numpy.typing import NDArray

def get_debye_function(
    variable: Union[float, NDArray[np.float64]],
    exponent: float = 1.5
) -> Union[float, NDArray[np.float64]]:
    """
    Compute the Debye function for a given variable and exponent.

    The Debye function is defined as:
        D(variable) = (3 / variable**exponent) * ∫₀^variable (t**exponent / (np.exp(t) - 1)) dt

    For wall loss coefficient calculations, the exponent is typically 1.5.

    Arguments:
        variable : The input value(s) for the Debye function.
        exponent : The exponent to use in the Debye function (default: 1.5).

    Returns:
        The value(s) of the Debye function for the given input.
    """
    from scipy.integrate import quad

    def integrand(t):
        return t**exponent / (np.exp(t) - 1)

    if np.isscalar(variable):
        if variable == 0:
            return 1.0
        integral, _ = quad(integrand, 0, variable)
        return (3 / variable**exponent) * integral
    else:
        # Vectorized for array input
        result = np.empty_like(variable, dtype=np.float64)
        for idx, val in np.ndenumerate(variable):
            if val == 0:
                result[idx] = 1.0
            else:
                integral, _ = quad(integrand, 0, val)
                result[idx] = (3 / val**exponent) * integral
        return result
