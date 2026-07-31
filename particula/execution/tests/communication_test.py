"""Tests for direct fixed-shape communication declaration validation."""

import importlib
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from particula.execution import Backend, Device
from particula.execution.communication import (
    CommunicationBoundaryMode,
    CommunicationDimensions,
    CommunicationMap,
    CommunicationMapForm,
    CommunicationRepresentation,
    CommunicationResourceShapes,
    CommunicationTransportMode,
    PrescribedVolumeDeclaration,
    validate_communication_declarations,
)

wp = pytest.importorskip(
    "warp", reason="communication validation requires Warp"
)


def _declarations(
    source=(0,),
    destination=(1,),
    enabled=(True,),
    rates=(1.0,),
    bounds=(2.0, 2.0),
):
    """Build valid CPU Warp communication declarations."""
    device = Device(Backend.WARP, "cpu")
    communication_map = CommunicationMap(
        CommunicationMapForm.ONE_DIMENSIONAL,
        CommunicationTransportMode.GAS,
        CommunicationBoundaryMode.CLOSED,
        CommunicationRepresentation.PARTICLE_RESOLVED,
        wp.array(source, dtype=wp.int32, device="cpu"),
        wp.array(destination, dtype=wp.int32, device="cpu"),
        wp.array(enabled, dtype=wp.bool, device="cpu"),
        wp.array(rates, dtype=wp.float64, device="cpu"),
    )
    volume = PrescribedVolumeDeclaration(
        wp.array((1.0, 2.0), dtype=wp.float64, device="cpu"),
        wp.array(bounds, dtype=wp.float64, device="cpu"),
    )
    shapes = CommunicationResourceShapes(
        CommunicationDimensions(2, 3, 1), device
    )
    return communication_map, volume, shapes


def test_valid_declarations_return_exact_identities():
    """A valid one-dimensional map returns every supplied record unchanged."""
    declarations = _declarations()
    assert validate_communication_declarations(*declarations) == declarations
    assert all(
        left is right
        for left, right in zip(
            validate_communication_declarations(*declarations),
            declarations,
            strict=True,
        )
    )


@pytest.mark.parametrize("form", list(CommunicationMapForm))
@pytest.mark.parametrize("mode", list(CommunicationTransportMode))
@pytest.mark.parametrize("boundary", list(CommunicationBoundaryMode))
def test_valid_map_forms_modes_and_boundaries(form, mode, boundary):
    """Both forms, modes, and declared boundaries have valid P1 metadata."""
    communication_map, volume, shapes = _declarations()
    object.__setattr__(communication_map, "form", form)
    object.__setattr__(communication_map, "transport_mode", mode)
    object.__setattr__(communication_map, "boundary_mode", boundary)
    validate_communication_declarations(communication_map, volume, shapes)


def test_empty_map_and_zero_boxes_are_valid_noops():
    """Empty edge lanes and canonical zero-box declarations need no writer."""
    device = Device(Backend.WARP, "cpu")
    communication_map = CommunicationMap(
        CommunicationMapForm.PAIR,
        CommunicationTransportMode.PARTICLE,
        CommunicationBoundaryMode.INFLOW,
        CommunicationRepresentation.PARTICLE_RESOLVED,
        wp.zeros(0, dtype=wp.int32, device="cpu"),
        wp.zeros(0, dtype=wp.int32, device="cpu"),
        wp.zeros(0, dtype=wp.bool, device="cpu"),
        wp.zeros(0, dtype=wp.float64, device="cpu"),
    )
    volume = PrescribedVolumeDeclaration(
        wp.zeros(0, dtype=wp.float64, device="cpu"),
        wp.zeros(0, dtype=wp.float64, device="cpu"),
    )
    validate_communication_declarations(
        communication_map,
        volume,
        CommunicationResourceShapes(CommunicationDimensions(0, 0, 0), device),
    )


@pytest.mark.parametrize(
    ("source", "destination", "rates", "match"),
    [
        ((-1,), (1,), (1.0,), "source_indices"),
        ((0,), (0,), (1.0,), "self edges"),
        ((0, 0), (1, 1), (1.0, 1.0), "unique"),
        ((0,), (1,), (-1.0,), "rates"),
    ],
)
def test_invalid_payloads_reject_without_mutation(
    source, destination, rates, match
):
    """Invalid domains and topology do not mutate any caller-owned payload."""
    declarations = _declarations(
        source, destination, (True,) * len(source), rates
    )
    arrays = (
        declarations[0].source_indices,
        declarations[0].destination_indices,
        declarations[0].enabled,
        declarations[0].rates,
        declarations[1].volumes,
        declarations[1].outbound_bounds,
    )
    wp.synchronize()
    before = [array.numpy().copy() for array in arrays]
    with pytest.raises(ValueError, match=match):
        validate_communication_declarations(*declarations)
    wp.synchronize()
    for array, expected in zip(arrays, before, strict=True):
        np.testing.assert_array_equal(array.numpy(), expected)


def test_outbound_bound_and_disabled_rate_rules():
    """Enabled lanes are summed, while disabled lanes still undergo domain scan."""
    with pytest.raises(ValueError, match="outbound totals"):
        validate_communication_declarations(*_declarations(rates=(3.0,)))
    declarations = _declarations(rates=(100.0,), enabled=(False,))
    validate_communication_declarations(*declarations)


def test_constructors_are_frozen_and_validate_exact_metadata():
    """Declarations reject invalid metadata and preserve frozen carriers."""
    with pytest.raises(TypeError, match="n_boxes"):
        CommunicationDimensions(True, 0, 0)
    with pytest.raises(TypeError, match="n_boxes"):
        CommunicationDimensions(np.int64(1), 0, 0)
    with pytest.raises(ValueError, match="n_particles"):
        CommunicationDimensions(0, -1, 0)
    dimensions = CommunicationDimensions(1, 0, 0)
    with pytest.raises(FrozenInstanceError):
        dimensions.n_boxes = 2
    with pytest.raises(TypeError, match="form"):
        CommunicationMap(
            "pair",
            CommunicationTransportMode.GAS,
            CommunicationBoundaryMode.CLOSED,
            CommunicationRepresentation.PARTICLE_RESOLVED,
            None,
            None,
            None,
            None,
        )


@pytest.mark.parametrize(
    ("field", "value", "error", "match"),
    [
        ("source_indices", None, TypeError, "source_indices"),
        (
            "source_indices",
            wp.array(((0,),), dtype=wp.int32, device="cpu"),
            ValueError,
            "rank 1",
        ),
        (
            "destination_indices",
            wp.array((1, 0), dtype=wp.int32, device="cpu"),
            ValueError,
            "destination_indices",
        ),
        (
            "rates",
            wp.array((1.0,), dtype=wp.float32, device="cpu"),
            TypeError,
            "rates",
        ),
        (
            "volumes",
            wp.array((1.0,), dtype=wp.float64, device="cpu"),
            ValueError,
            "volumes",
        ),
    ],
)
def test_schema_validation_rejects_array_type_rank_dtype_and_shape(
    field, value, error, match
):
    """Each payload lane must retain its exact fixed-capacity Warp schema."""
    communication_map, volume, shapes = _declarations()
    target = volume if field == "volumes" else communication_map
    object.__setattr__(target, field, value)

    with pytest.raises(error, match=match):
        validate_communication_declarations(communication_map, volume, shapes)


def test_pair_map_permits_arbitrary_and_reverse_directed_edges():
    """Pair maps accept distinct arbitrary boxes and reverse edge directions."""
    device = Device(Backend.WARP, "cpu")
    communication_map = CommunicationMap(
        CommunicationMapForm.PAIR,
        CommunicationTransportMode.PARTICLE,
        CommunicationBoundaryMode.OUTFLOW,
        CommunicationRepresentation.PARTICLE_RESOLVED,
        wp.array((0, 2), dtype=wp.int32, device="cpu"),
        wp.array((2, 0), dtype=wp.int32, device="cpu"),
        wp.array((True, True), dtype=wp.bool, device="cpu"),
        wp.array((1.0, 1.0), dtype=wp.float64, device="cpu"),
    )
    volume = PrescribedVolumeDeclaration(
        wp.array((1.0, 1.0, 1.0), dtype=wp.float64, device="cpu"),
        wp.array((1.0, 0.0, 1.0), dtype=wp.float64, device="cpu"),
    )
    shapes = CommunicationResourceShapes(
        CommunicationDimensions(3, 0, 0), device
    )

    assert validate_communication_declarations(
        communication_map, volume, shapes
    ) == (communication_map, volume, shapes)


@pytest.mark.parametrize(
    ("field", "values", "match"),
    [
        ("rates", (np.nan,), "rates"),
        ("rates", (np.inf,), "rates"),
        ("volumes", (0.0, 2.0), "volumes"),
        ("volumes", (np.inf, 2.0), "volumes"),
        ("outbound_bounds", (-1.0, 2.0), "outbound_bounds"),
        ("outbound_bounds", (np.nan, 2.0), "outbound_bounds"),
    ],
)
def test_disabled_lanes_still_validate_physical_payloads(field, values, match):
    """Disabled edges cannot hide invalid edge or box physical values."""
    communication_map, volume, shapes = _declarations(enabled=(False,))
    if field == "rates":
        object.__setattr__(
            communication_map,
            field,
            wp.array(values, dtype=wp.float64, device="cpu"),
        )
    else:
        object.__setattr__(
            volume,
            field,
            wp.array(values, dtype=wp.float64, device="cpu"),
        )

    with pytest.raises(ValueError, match=match):
        validate_communication_declarations(communication_map, volume, shapes)


def test_aliasing_float_payload_storage_is_rejected_before_domain_scans():
    """Overlapping nonempty float payload lanes fail the aliasing stage."""
    communication_map, volume, shapes = _declarations(
        source=(0, 1),
        destination=(1, 0),
        enabled=(True, True),
        rates=(1.0, 1.0),
        bounds=(2.0, 2.0),
    )
    object.__setattr__(communication_map, "rates", volume.volumes)

    with pytest.raises(ValueError, match="rates and volumes must not alias"):
        validate_communication_declarations(communication_map, volume, shapes)


def test_outbound_total_accepts_equal_bound_and_rejects_overflow():
    """Enabled rates use a strict per-source bound with finite-safe totals."""
    accepted = _declarations(
        source=(0, 0),
        destination=(1, 1),
        enabled=(True, False),
        rates=(2.0, 1.0),
    )
    validate_communication_declarations(*accepted)

    overdraw_map = CommunicationMap(
        CommunicationMapForm.PAIR,
        CommunicationTransportMode.GAS,
        CommunicationBoundaryMode.CLOSED,
        CommunicationRepresentation.PARTICLE_RESOLVED,
        wp.array((0, 0), dtype=wp.int32, device="cpu"),
        wp.array((1, 2), dtype=wp.int32, device="cpu"),
        wp.array((True, True), dtype=wp.bool, device="cpu"),
        wp.array((1.0e308, 1.0e308), dtype=wp.float64, device="cpu"),
    )
    overdraw_volume = PrescribedVolumeDeclaration(
        wp.array((1.0, 1.0, 1.0), dtype=wp.float64, device="cpu"),
        wp.array(
            (np.finfo(np.float64).max, 2.0, 2.0),
            dtype=wp.float64,
            device="cpu",
        ),
    )
    overdraw_shapes = CommunicationResourceShapes(
        CommunicationDimensions(3, 0, 0), Device(Backend.WARP, "cpu")
    )
    with pytest.raises(ValueError, match="outbound totals"):
        validate_communication_declarations(
            overdraw_map, overdraw_volume, overdraw_shapes
        )


def test_bypassed_metadata_and_package_exports_fail_closed():
    """Validator rechecks bypassed metadata and keeps declarations direct-only."""
    communication_map, volume, shapes = _declarations()
    object.__setattr__(shapes, "dimensions", "invalid")
    with pytest.raises(TypeError, match="dimensions"):
        validate_communication_declarations(communication_map, volume, shapes)

    module = importlib.import_module("particula.execution.communication")
    execution = importlib.import_module("particula.execution")
    assert module.validate_communication_declarations is (
        validate_communication_declarations
    )
    assert "validate_communication_declarations" not in execution.__all__
