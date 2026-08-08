"""Tests for the direct-only fixed-shape communication-map preflight."""

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest


def _communication() -> Any:
    """Import the Warp-dependent boundary only when Warp is installed."""
    pytest.importorskip("warp")
    import particula.execution.communication as communication

    return communication


def _configuration(
    communication: Any,
    *,
    form: Any = None,
    mode: Any = None,
    source: object | None = None,
    destination: object | None = None,
    enabled: object | None = None,
    rates: object | None = None,
    volumes: object | None = None,
) -> tuple[Any, Any]:
    """Build an independent canonical three-box two-edge configuration."""
    wp = pytest.importorskip("warp")
    form = (
        form
        if form is not None
        else communication.CommunicationMapForm.ONE_DIMENSIONAL
    )
    mode = (
        mode
        if mode is not None
        else communication.CommunicationTransportMode.GAS
    )
    source = (
        source
        if source is not None
        else wp.array([0, 1], dtype=wp.int32, device="cpu")
    )
    destination = (
        destination
        if destination is not None
        else wp.array([1, 2], dtype=wp.int32, device="cpu")
    )
    enabled = (
        enabled
        if enabled is not None
        else wp.array([1, 1], dtype=wp.int32, device="cpu")
    )
    rates = (
        rates
        if rates is not None
        else wp.array([1.0, 2.0], dtype=wp.float64, device="cpu")
    )
    map_data = communication.CommunicationMap(
        form, mode, 2, source, destination, enabled, rates
    )
    configuration = communication.CommunicationConfiguration(
        map_data,
        communication.PrescribedVolumeUpdate(volumes),
        (
            communication.CommunicationResourceShape(
                "edge_rates",
                wp.float64,
                communication.CommunicationShapeKind.E,
            ),
        ),
    )
    from particula.execution.gpu_session import ResidentDimensions

    return configuration, ResidentDimensions(3, 2, 1)


def test_carriers_are_frozen_and_validate_cheap_metadata() -> None:
    """Typed immutable declarations retain caller payload identities."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    resource = communication.CommunicationResourceShape(
        "edges", wp.float64, communication.CommunicationShapeKind.E
    )
    assert resource.role == "edges"
    with pytest.raises(FrozenInstanceError):
        resource.role = "other"  # type: ignore[misc]
    with pytest.raises(ValueError, match="nonempty"):
        communication.CommunicationResourceShape(
            " ", wp.float64, communication.CommunicationShapeKind.E
        )
    with pytest.raises(TypeError, match="integral"):
        communication.CommunicationMap(
            communication.CommunicationMapForm.ONE_DIMENSIONAL,
            communication.CommunicationTransportMode.GAS,
            True,
            object(),
            object(),
            object(),
            object(),
        )


def test_carrier_metadata_rejects_invalid_enum_nested_and_roles() -> None:
    """Carrier construction rejects malformed metadata before Warp preflight."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    with pytest.raises(TypeError, match="shape_kind"):
        communication.CommunicationResourceShape("edges", wp.float64, "e")
    with pytest.raises(TypeError, match="map form"):
        communication.CommunicationMap(
            "one_dimensional",
            communication.CommunicationTransportMode.GAS,
            0,
            object(),
            object(),
            object(),
            object(),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        communication.CommunicationMap(
            communication.CommunicationMapForm.ONE_DIMENSIONAL,
            communication.CommunicationTransportMode.GAS,
            -1,
            object(),
            object(),
            object(),
            object(),
        )
    map_data = communication.CommunicationMap(
        communication.CommunicationMapForm.ONE_DIMENSIONAL,
        communication.CommunicationTransportMode.GAS,
        0,
        object(),
        object(),
        object(),
        object(),
    )
    role = communication.CommunicationResourceShape(
        "edges", wp.float64, communication.CommunicationShapeKind.E
    )
    with pytest.raises(ValueError, match="roles must be unique"):
        communication.CommunicationConfiguration(
            map_data,
            communication.PrescribedVolumeUpdate(None),
            (role, role),
        )


@pytest.mark.warp
@pytest.mark.parametrize(
    "form",
    [
        "one_dimensional",
        "arbitrary_pairs",
    ],
)
@pytest.mark.parametrize("mode", ["gas", "particles", "gas_and_particles"])
def test_valid_forms_and_modes_return_exact_configuration(
    form: str, mode: str
) -> None:
    """Every form and transport declaration accepts an independent valid map."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    source = wp.array([0, 1], dtype=wp.int32, device="cpu")
    destination_values = [1, 2] if form == "one_dimensional" else [2, 0]
    destination = wp.array(destination_values, dtype=wp.int32, device="cpu")
    configuration, dimensions = _configuration(
        communication,
        form=communication.CommunicationMapForm(form),
        mode=communication.CommunicationTransportMode(mode),
        source=source,
        destination=destination,
        volumes=wp.array([1.0, 2.0, 3.0], dtype=wp.float64, device="cpu"),
    )

    result = communication.validate_communication_configuration(
        configuration, dimensions, wp.get_device("cpu")
    )

    assert result is configuration
    assert result.communication_map.source_boxes is source
    assert result.communication_map.destination_boxes is destination


@pytest.mark.warp
@pytest.mark.parametrize(
    ("field", "values", "match"),
    [
        ("enabled", [2, 1], "enabled values"),
        ("rates", [np.nan, 1.0], "rates values"),
    ],
)
def test_domain_validation_precedes_topology(
    field: str, values: list[float], match: str
) -> None:
    """Payload domains reject before an otherwise-invalid enabled endpoint."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    kwargs: dict[str, object] = {
        "source": wp.array([3, 1], dtype=wp.int32, device="cpu"),
    }
    dtype = wp.int32 if field == "enabled" else wp.float64
    kwargs[field] = wp.array(values, dtype=dtype, device="cpu")
    configuration, dimensions = _configuration(communication, **kwargs)

    with pytest.raises(ValueError, match=match):
        communication.validate_communication_configuration(
            configuration, dimensions, wp.get_device("cpu")
        )


@pytest.mark.warp
@pytest.mark.parametrize(
    ("source_values", "destination_values", "match"),
    [
        ([3, 1], [1, 2], "valid distinct topology"),
        ([0, 1], [0, 2], "valid distinct topology"),
        ([0, 0], [2, 2], "valid distinct topology"),
        ([0, 1], [2, 0], "valid distinct topology"),
    ],
)
def test_enabled_topology_rejections(
    source_values: list[int], destination_values: list[int], match: str
) -> None:
    """Bounds, self edges, duplicates, and one-dimensional gaps reject."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    configuration, dimensions = _configuration(
        communication,
        source=wp.array(source_values, dtype=wp.int32, device="cpu"),
        destination=wp.array(destination_values, dtype=wp.int32, device="cpu"),
    )
    with pytest.raises(ValueError, match=match):
        communication.validate_communication_configuration(
            configuration, dimensions, wp.get_device("cpu")
        )


@pytest.mark.warp
def test_duplicate_directed_pair_and_disabled_padding() -> None:
    """Reject duplicate pairs but ignore disabled endpoint padding."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    duplicate, dimensions = _configuration(
        communication,
        form=communication.CommunicationMapForm.ARBITRARY_PAIRS,
        source=wp.array([0, 0], dtype=wp.int32, device="cpu"),
        destination=wp.array([2, 2], dtype=wp.int32, device="cpu"),
    )
    with pytest.raises(ValueError, match="directed edge pairs"):
        communication.validate_communication_configuration(
            duplicate, dimensions, wp.get_device("cpu")
        )

    padded, dimensions = _configuration(
        communication,
        source=wp.array([-1, -1], dtype=wp.int32, device="cpu"),
        destination=wp.array([99, 99], dtype=wp.int32, device="cpu"),
        enabled=wp.array([0, 0], dtype=wp.int32, device="cpu"),
    )
    assert (
        communication.validate_communication_configuration(
            padded, dimensions, wp.get_device("cpu")
        )
        is padded
    )


@pytest.mark.warp
def test_schema_alias_and_empty_preflight() -> None:
    """Arrays need independent storage, while valid empty maps are no-ops."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    shared = wp.array([0, 1], dtype=wp.int32, device="cpu")
    configuration, dimensions = _configuration(communication, source=shared)
    object.__setattr__(
        configuration.communication_map, "destination_boxes", shared
    )
    with pytest.raises(ValueError, match="must not alias"):
        communication.validate_communication_configuration(
            configuration, dimensions, wp.get_device("cpu")
        )

    empty_map = communication.CommunicationMap(
        communication.CommunicationMapForm.ONE_DIMENSIONAL,
        communication.CommunicationTransportMode.GAS,
        0,
        wp.empty(0, dtype=wp.int32, device="cpu"),
        wp.empty(0, dtype=wp.int32, device="cpu"),
        wp.empty(0, dtype=wp.int32, device="cpu"),
        wp.empty(0, dtype=wp.float64, device="cpu"),
    )
    empty = communication.CommunicationConfiguration(
        empty_map, communication.PrescribedVolumeUpdate(None), ()
    )
    from particula.execution.gpu_session import ResidentDimensions

    assert (
        communication.validate_communication_configuration(
            empty, ResidentDimensions(0, 0, 0), wp.get_device("cpu")
        )
        is empty
    )


@pytest.mark.warp
@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("source_boxes", object(), "source_boxes must be a Warp array"),
        (
            "source_boxes",
            "rank",
            "source_boxes must have rank 1",
        ),
        (
            "source_boxes",
            "shape",
            "source_boxes must have shape (2,)",
        ),
        (
            "source_boxes",
            "dtype",
            "source_boxes must have dtype",
        ),
        (
            "source_boxes",
            "null_pointer",
            "source_boxes must have a valid pointer",
        ),
    ],
)
def test_source_schema_preflight_rejects_before_payload_scans(
    field: str,
    value: object,
    match: str,
) -> None:
    """Source schema failures precede all payload scans and writer activity."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    replacement: object = value
    if value == "rank":
        replacement = wp.array([[0, 1]], dtype=wp.int32, device="cpu")
    elif value == "shape":
        replacement = wp.array([0], dtype=wp.int32, device="cpu")
    elif value == "dtype":
        replacement = wp.array([0.0, 1.0], dtype=wp.float64, device="cpu")
    elif value == "null_pointer":
        replacement = wp.array(
            ptr=0,
            capacity=8,
            dtype=wp.int32,
            shape=(2,),
            strides=(4,),
            device="cpu",
            copy=False,
        )
    configuration, dimensions = _configuration(communication)
    object.__setattr__(configuration.communication_map, field, replacement)

    with pytest.raises(ValueError, match=match):
        communication.validate_communication_configuration(
            configuration, dimensions, wp.get_device("cpu")
        )


@pytest.mark.warp
@pytest.mark.parametrize("volume", [[0.0, 2.0, 3.0], [np.nan, 2.0, 3.0]])
def test_volume_domain_is_scanned_before_topology(
    volume: list[float],
) -> None:
    """Invalid prescribed volumes reject before invalid enabled endpoints."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    configuration, dimensions = _configuration(
        communication,
        source=wp.array([3, 1], dtype=wp.int32, device="cpu"),
        volumes=wp.array(volume, dtype=wp.float64, device="cpu"),
    )

    with pytest.raises(ValueError, match="final_volumes values"):
        communication.validate_communication_configuration(
            configuration, dimensions, wp.get_device("cpu")
        )


@pytest.mark.warp
def test_validation_is_read_only_for_all_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful preflight neither copies nor changes caller-owned payloads."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    volumes = wp.array([1.0, 2.0, 3.0], dtype=wp.float64, device="cpu")
    configuration, dimensions = _configuration(communication, volumes=volumes)
    map_data = configuration.communication_map
    payloads = (
        map_data.source_boxes,
        map_data.destination_boxes,
        map_data.enabled,
        map_data.rates,
        volumes,
    )
    snapshots = [payload.numpy().copy() for payload in payloads]
    monkeypatch.setattr(
        communication.wp,
        "copy",
        lambda *_args, **_kwargs: pytest.fail("P1 must not copy payloads"),
    )

    assert (
        communication.validate_communication_configuration(
            configuration, dimensions, wp.get_device("cpu")
        )
        is configuration
    )
    for payload, snapshot in zip(payloads, snapshots, strict=True):
        npt.assert_array_equal(payload.numpy(), snapshot)


@pytest.mark.warp
def test_private_range_and_duplicate_helpers_cover_edge_cases() -> None:
    """Private range overlap and hash sizing preserve bounded-map invariants."""
    communication = _communication()
    assert communication._overlaps((4, 8), (8, 12)) is False
    assert communication._overlaps((4, 9), (8, 12)) is True
    assert communication._overlaps(None, (8, 12)) is False
    assert communication._duplicate_table_size(0) == 2
    assert communication._duplicate_table_size(3) == 8


def test_communication_remains_unexported_and_has_no_overdraw_input() -> None:
    """Package import neither imports this concrete module nor claims P3 work."""
    root = Path(__file__).parents[3]
    environment = os.environ | {"PYTHONPATH": str(root)}
    script = """
import sys
import particula.execution as execution
assert 'CommunicationConfiguration' not in execution.__all__
assert 'particula.execution.communication' not in sys.modules
assert 'warp' not in sys.modules
"""
    completed = subprocess.run(  # noqa: S603 -- fixed interpreter and script
        [sys.executable, "-Werror", "-c", script],
        cwd=root,
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    communication = _communication()
    validator_docstring = (
        communication.validate_communication_configuration.__doc__
    )
    assert "source inventory" in validator_docstring
