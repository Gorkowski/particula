"""
Tests for Converting Units
"""

import unittest
from particula.util.convert_units import get_unit_conversion

# flake8: noqa
try:
    import pint  # pylint: disable=unused-import
except ImportError:
    IS_PINT_AVAILABLE = False
else:
    IS_PINT_AVAILABLE = True


class TestUnitConversion(unittest.TestCase):
    """
    Test for Unit Conversion
    """

    def setUp(self) -> None:
        """
        Setup the test environment
        """

    def test_import_warning(self) -> None:
        """
        Test for import warning if pint is not installed
        """
        if not IS_PINT_AVAILABLE:
            with self.assertRaises(ImportError):
                get_unit_conversion("degC", "degF")
        else:
            self.skipTest("Pint installed. Skipping import warning test.")

    def test_example_conversion(self) -> None:
        """
        Test for example conversion when pint is installed
        """
        if IS_PINT_AVAILABLE:
            result = get_unit_conversion("ug/m^3", "kg/m^3")
            self.assertAlmostEqual(result, 1e-9)
        else:
            self.skipTest("Pint not installed. Skipping conversion test.")

    def test_temperature_conversion(self) -> None:
        """
        Test temperature conversion from Celsius to Fahrenheit
        """
        if IS_PINT_AVAILABLE:
            result = get_unit_conversion("degC", "degF", value=0)
            self.assertAlmostEqual(result, 32.0, places=6)
        else:
            self.skipTest("Pint not installed. Skipping temperature conversion test.")
