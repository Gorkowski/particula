"""
Test Special Functions
"""

import numpy as np
from particula.next.particles.properties.special_functions import (
    debye_function,
)


def test_debye_function_single_float():
    result = debye_function(1.0)
    assert np.isclose(result, 0.8414056604369606)

    result = debye_function(1.0, n=2)
    assert np.isclose(result, 0.6007582206816492)


def test_debye_function_numpy_array():
    input_array = np.array([1.0, 2.0, 3.0])
    expected_output = np.array([0.84140566, 0.42278434, 0.28784241])
    result = debye_function(input_array)
    assert np.allclose(result, expected_output)


def test_debye_function_with_integration_points():
    result = debye_function(1.0, integration_points=100)
    assert np.isclose(result, 0.8414056604369606)


def test_debye_function_with_custom_n_and_integration_points():
    result = debye_function(1.0, integration_points=100, n=2)
    assert np.isclose(result, 0.6007582206816492)
