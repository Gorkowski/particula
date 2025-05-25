"""Tests for EnvironmentalConditions and its fluent Builder."""


import pytest

from path import (  # adapt the import path
    EnvironmentalConditions,
    EnvironmentalConditionsBuilder,
)


# ----------------------------------------------------------------------
# 1 · Default behaviour
# ----------------------------------------------------------------------
def test_builder_returns_defaults():
    """Empty builder → identical to a bare dataclass instantiation."""
    expected = EnvironmentalConditions()
    result = EnvironmentalConditionsBuilder().build()
    assert result == expected


# ----------------------------------------------------------------------
# 2 · Single-field overrides (parametrised)
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", 310.0),
        ("pressure", 90_000.0),
        ("mass_accommodation", 0.8),
        ("dynamic_viscosity", 2.2e-5),
        ("diffusion_coefficient", 3.1e-5),
        ("time_step", 5.0),
        ("simulation_volume", 2.0e-6),
    ],
)
def test_single_override(field: str, value: float):
    """Only the specified field should differ from defaults."""
    builder = EnvironmentalConditionsBuilder()
    getattr(builder, field)(value)  # call the fluent setter
    result = builder.build()
    default = EnvironmentalConditions()
    assert getattr(result, field) == value
    # all other attributes remain at default
    for f in default.__dataclass_fields__:
        if f != field:
            assert getattr(result, f) == getattr(default, f)


# ----------------------------------------------------------------------
# 3 · Chained setters
# ----------------------------------------------------------------------
def test_chained_overrides():
    """Multiple fluent calls should accumulate correctly."""
    env = (
        EnvironmentalConditionsBuilder()
        .temperature(305.0)
        .pressure(95_000.0)
        .dynamic_viscosity(1.9e-5)
        .build()
    )
    assert env.temperature == 305.0
    assert env.pressure == 95_000.0
    assert env.dynamic_viscosity == 1.9e-5
    # unchanged fields stay at default
    assert env.time_step == EnvironmentalConditions().time_step


# ----------------------------------------------------------------------
# 4 · Immutability of the dataclass
# ----------------------------------------------------------------------
def test_dataclass_is_frozen():
    env = EnvironmentalConditionsBuilder().build()
    with pytest.raises(dataclasses.FrozenInstanceError):
        env.temperature = 999.0  # type: ignore[misc]


# ----------------------------------------------------------------------
# 5 · Builder reuse safety
# ----------------------------------------------------------------------
def test_builder_reuse_creates_independent_objects():
    """State should not leak between successive .build() calls."""
    builder = (
        EnvironmentalConditionsBuilder().temperature(300.0).pressure(95000.0)
    )
    env1 = builder.build()
    env2 = builder.temperature(310.0).build()
    assert env1.temperature == 300.0  # unchanged
    assert env2.temperature == 310.0
    assert env1.pressure == env2.pressure == 95000.0
