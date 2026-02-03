"""Tests for the GasData dataclass."""

import numpy as np
import numpy.testing as npt
import pytest
from particula.gas.gas_data import GasData, from_species, to_species
from particula.gas.species import GasSpecies
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)
from particula.util.constants import AVOGADRO_NUMBER


class TestFromSpecies:
    """Tests for from_species conversion utility."""

    def test_from_species_single_species_unit_conversion(self) -> None:
        """Single species converts kg/m^3 to molecules/m^3 with tiling."""
        strategy = ConstantVaporPressureStrategy(2330.0)
        species = GasSpecies(
            name="Water",
            molar_mass=0.018,
            vapor_pressure_strategy=strategy,
            partitioning=True,
            concentration=1e-6,
        )

        data = from_species(species)

        expected_number = (1e-6 / 0.018) * AVOGADRO_NUMBER
        assert data.n_boxes == 1
        assert data.n_species == 1
        npt.assert_allclose(data.concentration, np.array([[expected_number]]))
        npt.assert_allclose(data.molar_mass, np.array([0.018]))
        np.testing.assert_array_equal(data.partitioning, np.array([True]))
        assert data.name == ["Water"]

    def test_from_species_multi_species_arrays(self) -> None:
        """Multi-species arrays preserve fields and convert units."""
        strategy = ConstantVaporPressureStrategy(0.0)
        base = GasSpecies(
            name="Water",
            molar_mass=0.018,
            vapor_pressure_strategy=strategy,
            partitioning=True,
            concentration=2e-6,
        )
        other = GasSpecies(
            name="Ammonia",
            molar_mass=0.017,
            vapor_pressure_strategy=strategy,
            partitioning=True,
            concentration=3e-6,
        )
        species = base + other

        data = from_species(species)

        expected = np.array(
            [
                (2e-6 / 0.018) * AVOGADRO_NUMBER,
                (3e-6 / 0.017) * AVOGADRO_NUMBER,
            ]
        )
        npt.assert_allclose(data.concentration, expected.reshape(1, 2))
        npt.assert_allclose(data.molar_mass, np.array([0.018, 0.017]))
        np.testing.assert_array_equal(data.partitioning, np.array([True, True]))
        assert data.name == ["Water", "Ammonia"]

    def test_from_species_n_boxes_replication(self) -> None:
        """Concentration replicates across boxes when n_boxes > 1."""
        species = GasSpecies(
            name="Water",
            molar_mass=0.018,
            vapor_pressure_strategy=ConstantVaporPressureStrategy(0.0),
            partitioning=False,
            concentration=1e-6,
        )

        data = from_species(species, n_boxes=3)

        expected = ((1e-6 / 0.018) * AVOGADRO_NUMBER) * np.ones((3, 1))
        npt.assert_allclose(data.concentration, expected)
        assert data.n_boxes == 3
        assert data.n_species == 1

    def test_from_species_invalid_n_boxes_raises(self) -> None:
        """n_boxes < 1 raises ValueError."""
        species = GasSpecies(
            name="Water",
            molar_mass=0.018,
            vapor_pressure_strategy=ConstantVaporPressureStrategy(0.0),
            partitioning=True,
            concentration=1e-6,
        )

        with pytest.raises(ValueError, match="n_boxes"):
            from_species(species, n_boxes=0)


class TestGasDataInstantiation:
    """Tests for valid GasData instantiation and accessors."""

    def test_valid_single_box(self) -> None:
        """Test valid instantiation with single box (n_boxes=1)."""
        gas = GasData(
            name=["Water", "Ammonia", "H2SO4"],
            molar_mass=np.array([0.018, 0.017, 0.098]),
            concentration=np.array([[1e15, 1e12, 1e10]]),
            partitioning=np.array([True, True, True]),
        )

        assert gas.n_boxes == 1
        assert gas.n_species == 3
        assert gas.concentration.shape == (1, 3)
        assert gas.molar_mass.shape == (3,)
        assert gas.partitioning.shape == (3,)

    def test_valid_multi_box(self) -> None:
        """Test valid instantiation with multiple boxes (n_boxes>1)."""
        gas = GasData(
            name=["Water", "Ammonia"],
            molar_mass=np.array([0.018, 0.017]),
            concentration=np.zeros((4, 2)),
            partitioning=np.array([True, False]),
        )

        assert gas.n_boxes == 4
        assert gas.n_species == 2
        assert gas.concentration.shape == (4, 2)
        assert gas.molar_mass.shape == (2,)
        assert gas.partitioning.shape == (2,)

    def test_valid_single_species(self) -> None:
        """Test edge case: single species (n_species=1)."""
        gas = GasData(
            name=["Water"],
            molar_mass=np.array([0.018]),
            concentration=np.array([[1e12], [2e12]]),
            partitioning=np.array([True]),
        )

        assert gas.n_boxes == 2
        assert gas.n_species == 1
        assert gas.concentration.shape == (2, 1)
        assert gas.molar_mass.shape == (1,)
        assert gas.partitioning.shape == (1,)


class TestGasDataValidation:
    """Tests for GasData validation errors."""

    def test_molar_mass_shape_mismatch(self) -> None:
        """Validation error when molar_mass has wrong shape."""
        with pytest.raises(ValueError, match="molar_mass shape"):
            GasData(
                name=["Water", "Ammonia"],
                molar_mass=np.array([[0.018, 0.017]]),  # wrong shape
                concentration=np.zeros((1, 2)),
                partitioning=np.array([True, True]),
            )

    def test_partitioning_shape_mismatch(self) -> None:
        """Validation error when partitioning has wrong shape."""
        with pytest.raises(ValueError, match="partitioning shape"):
            GasData(
                name=["Water", "Ammonia"],
                molar_mass=np.array([0.018, 0.017]),
                concentration=np.zeros((1, 2)),
                partitioning=np.array([[True, False]]),  # wrong shape
            )

    def test_partitioning_casts_to_bool(self) -> None:
        """Partitioning is coerced to boolean dtype."""
        gas = GasData(
            name=["Water", "Ammonia"],
            molar_mass=np.array([0.018, 0.017]),
            concentration=np.zeros((1, 2)),
            partitioning=np.array([1, 0]),  # convertible to bool
        )

        assert gas.partitioning.dtype == np.bool_
        np.testing.assert_array_equal(gas.partitioning, np.array([True, False]))

    def test_concentration_not_2d(self) -> None:
        """Validation error when concentration is not 2D."""
        with pytest.raises(ValueError, match="must be 2D"):
            GasData(
                name=["Water", "Ammonia"],
                molar_mass=np.array([0.018, 0.017]),
                concentration=np.array([1e15, 1e12]),  # 1D instead of 2D
                partitioning=np.array([True, True]),
            )

    def test_concentration_n_species_mismatch(self) -> None:
        """Validation error when concentration n_species doesn't match names."""
        with pytest.raises(ValueError, match="n_species dimension"):
            GasData(
                name=["Water", "Ammonia", "H2SO4"],
                molar_mass=np.array([0.018, 0.017, 0.098]),
                concentration=np.zeros((2, 2)),  # width 2 vs 3 names
                partitioning=np.array([True, True, True]),
            )

    def test_empty_name_raises(self) -> None:
        """Empty name list raises ValueError."""
        with pytest.raises(ValueError, match="at least one species"):
            GasData(
                name=[],
                molar_mass=np.array([], dtype=np.float64),
                concentration=np.zeros((1, 0)),
                partitioning=np.array([], dtype=bool),
            )


class TestToSpecies:
    """Tests for to_species conversion utility."""

    def test_to_species_single_species(self) -> None:
        """Converts GasData back to single-species GasSpecies."""
        strategy = ConstantVaporPressureStrategy(0.0)
        data = GasData(
            name=["Water"],
            molar_mass=np.array([0.018]),
            concentration=np.array([[1e20]]),
            partitioning=np.array([True]),
        )

        species = to_species(data, [strategy])

        expected_mass = (1e20 * 0.018) / AVOGADRO_NUMBER
        assert isinstance(species, GasSpecies)
        assert species.get_name() == "Water"
        assert species.get_molar_mass() == 0.018
        assert species.get_partitioning() is True
        npt.assert_allclose(species.get_concentration(), expected_mass)

    def test_to_species_multi_species_order_and_values(self) -> None:
        """Multi-species conversion preserves order and values."""
        strategies = [ConstantVaporPressureStrategy(val) for val in [1.0, 2.0]]
        data = GasData(
            name=["Water", "Ammonia"],
            molar_mass=np.array([0.018, 0.017]),
            concentration=np.array([[1e20, 2e20]]),
            partitioning=np.array([False, False]),
        )

        species = to_species(data, strategies)

        expected_mass = np.array([1e20, 2e20]) * np.array([0.018, 0.017])
        expected_mass = expected_mass / AVOGADRO_NUMBER
        assert isinstance(species, GasSpecies)
        assert list(species.get_name()) == ["Water", "Ammonia"]
        npt.assert_allclose(species.get_concentration(), expected_mass)
        assert species.get_partitioning() is False

    def test_to_species_box_index_selection(self) -> None:
        """Selects the correct box via box_index."""
        strategies = [ConstantVaporPressureStrategy(val) for val in [1.0]]
        data = GasData(
            name=["Water"],
            molar_mass=np.array([0.018]),
            concentration=np.array([[1e19], [3e19]]),
            partitioning=np.array([True]),
        )

        species = to_species(data, strategies, box_index=1)

        expected_mass = (3e19 * 0.018) / AVOGADRO_NUMBER
        npt.assert_allclose(species.get_concentration(), expected_mass)

    def test_to_species_strategy_length_mismatch_raises(self) -> None:
        """ValueError when strategy list length mismatches n_species."""
        data = GasData(
            name=["Water", "Ammonia"],
            molar_mass=np.array([0.018, 0.017]),
            concentration=np.array([[1e20, 2e20]]),
            partitioning=np.array([True, True]),
        )

        with pytest.raises(ValueError, match="vapor_pressure_strategies"):
            to_species(data, [ConstantVaporPressureStrategy(1.0)])

    def test_to_species_box_index_out_of_range_raises(self) -> None:
        """IndexError when box_index is invalid."""
        data = GasData(
            name=["Water"],
            molar_mass=np.array([0.018]),
            concentration=np.array([[1e20]]),
            partitioning=np.array([True]),
        )

        with pytest.raises(IndexError, match="box_index"):
            to_species(data, [ConstantVaporPressureStrategy(1.0)], box_index=2)

    def test_to_species_mixed_partitioning_raises(self) -> None:
        """ValueError when partitioning differs across species."""
        data = GasData(
            name=["Water", "Ammonia"],
            molar_mass=np.array([0.018, 0.017]),
            concentration=np.array([[1e20, 2e20]]),
            partitioning=np.array([True, False]),
        )

        strategies = [ConstantVaporPressureStrategy(val) for val in [1.0, 2.0]]
        with pytest.raises(ValueError, match="partitioning"):
            to_species(data, strategies)


class TestGasDataProperties:
    """Tests for GasData properties."""

    def test_n_boxes_property(self) -> None:
        """n_boxes returns correct value."""
        gas = GasData(
            name=["Water", "Ammonia"],
            molar_mass=np.array([0.018, 0.017]),
            concentration=np.zeros((5, 2)),
            partitioning=np.array([True, True]),
        )

        assert gas.n_boxes == 5

    def test_n_species_property(self) -> None:
        """n_species returns correct value."""
        gas = GasData(
            name=["Water", "Ammonia", "H2SO4"],
            molar_mass=np.array([0.018, 0.017, 0.098]),
            concentration=np.zeros((3, 3)),
            partitioning=np.array([True, True, True]),
        )

        assert gas.n_species == 3


class TestGasDataRoundTrip:
    """Round-trip conversion GasSpecies -> GasData -> GasSpecies."""

    def test_round_trip_preserves_data_and_partitioning(self) -> None:
        """Round-trip preserves concentration within tolerance and flags."""
        strategies = [ConstantVaporPressureStrategy(val) for val in [1.0, 2.0]]
        base = GasSpecies(
            name="Water",
            molar_mass=0.018,
            vapor_pressure_strategy=strategies[0],
            partitioning=True,
            concentration=2e-6,
        )
        other = GasSpecies(
            name="Ammonia",
            molar_mass=0.017,
            vapor_pressure_strategy=strategies[1],
            partitioning=True,
            concentration=3e-6,
        )
        species = base + other

        data = from_species(species, n_boxes=2)
        reconstructed = to_species(data, strategies, box_index=1)

        expected_mass = np.array([2e-6, 3e-6])
        npt.assert_allclose(
            reconstructed.get_concentration(), expected_mass, rtol=1e-12
        )
        assert reconstructed.get_partitioning() is True
        assert list(reconstructed.get_name()) == ["Water", "Ammonia"]


class TestGasDataCopy:
    """Tests for GasData copy method."""

    def test_copy_creates_independent_arrays(self) -> None:
        """copy() returns new arrays that do not share memory."""
        gas = GasData(
            name=["Water", "Ammonia"],
            molar_mass=np.array([0.018, 0.017]),
            concentration=np.array([[1e15, 1e12], [2e15, 2e12]]),
            partitioning=np.array([True, False]),
        )

        gas_copy = gas.copy()

        assert not np.shares_memory(gas.molar_mass, gas_copy.molar_mass)
        assert not np.shares_memory(gas.concentration, gas_copy.concentration)
        assert not np.shares_memory(gas.partitioning, gas_copy.partitioning)

    def test_copy_preserves_values(self) -> None:
        """copy() preserves all values."""
        gas = GasData(
            name=["Water", "Ammonia", "H2SO4"],
            molar_mass=np.array([0.018, 0.017, 0.098]),
            concentration=np.array([[1e15, 1e12, 1e10]]),
            partitioning=np.array([True, False, True]),
        )

        gas_copy = gas.copy()

        npt.assert_allclose(gas_copy.molar_mass, gas.molar_mass)
        npt.assert_allclose(gas_copy.concentration, gas.concentration)
        np.testing.assert_array_equal(gas_copy.partitioning, gas.partitioning)

    def test_copy_independent_name_list(self) -> None:
        """copy() produces independent name list."""
        gas = GasData(
            name=["Water", "Ammonia"],
            molar_mass=np.array([0.018, 0.017]),
            concentration=np.array([[1e15, 1e12]]),
            partitioning=np.array([True, False]),
        )

        gas_copy = gas.copy()
        gas.name[0] = "Changed"

        assert gas_copy.name[0] == "Water"
