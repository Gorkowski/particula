"""Tests for GasDataBuilder covering conversions, shapes, and validation."""

import numpy as np
import pytest
from particula.gas.gas_data import GasData
from particula.gas.gas_data_builder import GasDataBuilder

pint = pytest.importorskip("pint")


class TestGasDataBuilderBasics:
    """Basic builds and fluent chaining."""

    def test_build_valid_single_box(self) -> None:
        """Builder creates valid GasData with unit conversion."""
        gas = (
            GasDataBuilder()
            .set_names(["Water", "Ammonia", "H2SO4"])
            .set_molar_mass([18.015, 17.031, 98.079], units="g/mol")
            .set_concentration([1e9, 1e6, 1e4], units="1/cm^3")
            .set_partitioning([True, True, True])
            .build()
        )

        assert isinstance(gas, GasData)
        assert gas.concentration.shape == (1, 3)
        np.testing.assert_allclose(
            gas.molar_mass, np.array([0.018015, 0.017031, 0.098079])
        )
        np.testing.assert_allclose(
            gas.concentration, np.array([[1e15, 1e12, 1e10]])
        )
        assert gas.partitioning.dtype == np.bool_

    def test_fluent_chaining_returns_self(self) -> None:
        """All setters return self for chaining."""
        builder = GasDataBuilder()
        assert builder.set_names(["Water"]) is builder
        assert builder.set_molar_mass([0.018]) is builder
        assert builder.set_concentration([1e15]) is builder
        assert builder.set_partitioning([True]) is builder
        assert builder.set_n_boxes(2) is builder


class TestGasDataBuilderUnitConversion:
    """Unit conversion coverage."""

    def test_molar_mass_g_mol_conversion(self) -> None:
        """Molar mass converted from g/mol to kg/mol."""
        gas = (
            GasDataBuilder()
            .set_names(["Water"])
            .set_molar_mass([18.015], units="g/mol")
            .set_concentration([1e15])
            .set_partitioning([True])
            .build()
        )
        np.testing.assert_allclose(gas.molar_mass, np.array([0.018015]))

    def test_concentration_cm3_conversion(self) -> None:
        """Concentration converted from 1/cm^3 to 1/m^3."""
        gas = (
            GasDataBuilder()
            .set_names(["Water"])
            .set_molar_mass([0.018], units="kg/mol")
            .set_concentration([1e3], units="1/cm^3")
            .set_partitioning([True])
            .build()
        )
        np.testing.assert_allclose(gas.concentration, np.array([[1e9]]))


class TestGasDataBuilderBatchDimension:
    """Automatic batch dimension handling."""

    def test_auto_batch_dimension_added_for_1d(self) -> None:
        """1D concentration gets expanded to (1, n_species)."""
        gas = (
            GasDataBuilder()
            .set_names(["Water", "Ammonia"])
            .set_molar_mass([0.018, 0.017])
            .set_concentration([1e15, 1e12])
            .set_partitioning([True, True])
            .build()
        )
        assert gas.concentration.shape == (1, 2)

    def test_two_dim_concentration_unchanged(self) -> None:
        """2D concentration remains unchanged."""
        gas = (
            GasDataBuilder()
            .set_names(["Water", "Ammonia"])
            .set_molar_mass([0.018, 0.017])
            .set_concentration([[1e15, 1e12], [2e15, 2e12]])
            .set_partitioning([True, True])
            .build()
        )
        assert gas.concentration.shape == (2, 2)

    def test_set_n_boxes_broadcasts_from_single_row(self) -> None:
        """set_n_boxes broadcasts single-row concentration to n_boxes."""
        gas = (
            GasDataBuilder()
            .set_n_boxes(3)
            .set_names(["Water", "Ammonia"])
            .set_molar_mass([0.018, 0.017])
            .set_concentration([1e15, 1e12])
            .set_partitioning([True, True])
            .build()
        )
        assert gas.concentration.shape == (3, 2)
        np.testing.assert_allclose(
            gas.concentration,
            np.array(
                [
                    [1e15, 1e12],
                    [1e15, 1e12],
                    [1e15, 1e12],
                ]
            ),
        )

    def test_set_n_boxes_mismatch_raises(self) -> None:
        """set_n_boxes raises when rows differ from requested n_boxes."""
        builder = (
            GasDataBuilder()
            .set_n_boxes(2)
            .set_names(["Water", "Ammonia"])
            .set_molar_mass([0.018, 0.017])
            .set_concentration([[1e15, 1e12], [2e15, 2e12]])
            .set_partitioning([True, True])
        )
        builder.set_concentration([[1e15, 1e12], [2e15, 2e12], [3e15, 3e12]])
        with pytest.raises(ValueError, match="concentration row count"):
            builder.build()


class TestGasDataBuilderValidation:
    """Validation errors for inputs and shapes."""

    def test_missing_required_fields_raise(self) -> None:
        """Missing any required field raises ValueError."""
        builder = GasDataBuilder()
        with pytest.raises(ValueError, match="names is required"):
            builder.build()

        builder = GasDataBuilder().set_names(["Water"])
        with pytest.raises(ValueError, match="molar_mass is required"):
            builder.build()

        builder = GasDataBuilder().set_names(["Water"]).set_molar_mass([0.018])
        with pytest.raises(ValueError, match="concentration is required"):
            builder.build()

        builder = (
            GasDataBuilder()
            .set_names(["Water"])
            .set_molar_mass([0.018])
            .set_concentration([1e15])
        )
        with pytest.raises(ValueError, match="partitioning is required"):
            builder.build()

    def test_negative_molar_mass_raises(self) -> None:
        """Non-positive molar mass raises ValueError."""
        builder = (
            GasDataBuilder()
            .set_names(["Water"])
            .set_concentration([1e15])
            .set_partitioning([True])
        )
        with pytest.raises(ValueError, match="positive"):
            builder.set_molar_mass([-1.0])

    def test_negative_concentration_raises(self) -> None:
        """Negative concentration raises ValueError."""
        builder = (
            GasDataBuilder()
            .set_names(["Water"])
            .set_molar_mass([0.018])
            .set_partitioning([True])
        )
        with pytest.raises(ValueError, match="non-negative"):
            builder.set_concentration([-1.0])

    def test_molar_mass_wrong_dimension_raises(self) -> None:
        """Non-1D molar_mass raises ValueError."""
        builder = GasDataBuilder().set_names(["Water"]).set_partitioning([True])
        with pytest.raises(ValueError, match="1D"):
            builder.set_molar_mass([[0.018]])

    def test_concentration_wrong_dimension_raises(self) -> None:
        """Concentration must be 1D or 2D."""
        builder = GasDataBuilder().set_names(["Water"]).set_partitioning([True])
        with pytest.raises(ValueError, match="1D or 2D"):
            builder.set_concentration(np.ones((1, 1, 1)))

    def test_shape_mismatch_vs_names_raises(self) -> None:
        """Shape mismatches against names raise ValueError."""
        builder = (
            GasDataBuilder()
            .set_names(["Water", "Ammonia"])
            .set_molar_mass([0.018, 0.017])
            .set_partitioning([True, True])
            .set_concentration([1e15, 1e12])
        )
        builder._concentration = np.array([[1e15]])  # type: ignore[attr-defined]
        with pytest.raises(ValueError, match="width must match names"):
            builder.build()

    def test_partitioning_shape_mismatch_raises(self) -> None:
        """Partitioning shape mismatch raises ValueError."""
        builder = (
            GasDataBuilder()
            .set_names(["Water", "Ammonia"])
            .set_molar_mass([0.018, 0.017])
            .set_concentration([1e15, 1e12])
            .set_partitioning([True, True, True])
        )
        with pytest.raises(ValueError, match="partitioning length"):
            builder.build()

    def test_empty_names_raises(self) -> None:
        """Empty names list raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            GasDataBuilder().set_names([])

    def test_invalid_units_raise(self) -> None:
        """Undefined units propagate pint errors."""
        builder = GasDataBuilder().set_names(["Water"]).set_partitioning([True])
        with pytest.raises(pint.errors.UndefinedUnitError):
            builder.set_molar_mass([1.0], units="invalid")
        with pytest.raises(pint.errors.UndefinedUnitError):
            builder.set_concentration([1.0], units="invalid")
