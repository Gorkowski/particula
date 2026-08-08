"""Declare and preflight fixed-shape resident communication maps.

This concrete-only P1 boundary validates caller-owned Warp communication maps
without reading resident primaries, copying payloads, or launching a writer.
It is deliberately absent from :mod:`particula.execution` exports.  Empty and
disabled maps remain write-free only after their applicable schema and domain
preflight succeeds.

Population-dependent outbound overdraw is intentionally deferred to P3, which
will own source inventories and time-step inputs.  P2 owns volume writers; P4
owns particle transport; and P5 owns resident binding and primary/sidecar
aliasing.  This module provides none of those operations, synchronization,
host conversion, fallback, or scheduling.
"""

# mypy: disable-error-code="valid-type, misc"

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Any, cast

import warp as wp

from particula.execution.gpu_session import ResidentDimensions


class CommunicationMapForm(str, Enum):
    """Declare the enabled-edge topology accepted by a communication map."""

    ONE_DIMENSIONAL = "one_dimensional"
    ARBITRARY_PAIRS = "arbitrary_pairs"


class CommunicationTransportMode(str, Enum):
    """Declare which resident payload class a later phase may transport."""

    GAS = "gas"
    PARTICLES = "particles"
    GAS_AND_PARTICLES = "gas_and_particles"


class CommunicationShapeKind(str, Enum):
    """Declare fixed resource-shape vocabulary without retaining arrays."""

    B = "b"
    E = "e"
    BE = "be"
    BS = "bs"
    BN = "bn"
    BNS = "bns"


@dataclass(frozen=True)
class CommunicationResourceShape:
    """Describe one named fixed-shape communication resource.

    Args:
        role: Nonempty, unpadded resource role.  This carrier owns no array.
        dtype: Declared resource dtype metadata, retained unchanged.
        shape_kind: Exact fixed-shape vocabulary value.
    """

    role: str
    dtype: object
    shape_kind: CommunicationShapeKind

    def __post_init__(self) -> None:
        """Perform cheap role and shape metadata validation."""
        if type(self.role) is not str:
            raise TypeError("resource role must be an exact str.")
        if not self.role or self.role != self.role.strip():
            raise ValueError("resource role must be nonempty and unpadded.")
        if type(self.shape_kind) is not CommunicationShapeKind:
            raise TypeError(
                "resource shape_kind must be CommunicationShapeKind."
            )


@dataclass(frozen=True, eq=False)
class CommunicationMap:
    """Retain fixed-capacity caller-owned map arrays by identity.

    ``source_boxes``, ``destination_boxes``, and ``enabled`` must later be
    same-device contiguous pointer-backed ``wp.int32`` arrays shaped ``(E,)``.
    ``rates`` must be a corresponding ``wp.float64`` array in 1/s.  Validation
    neither normalizes nor mutates these arrays.
    """

    form: CommunicationMapForm
    transport_mode: CommunicationTransportMode
    edge_capacity: int
    source_boxes: object
    destination_boxes: object
    enabled: object
    rates: object

    def __post_init__(self) -> None:
        """Validate only fixed-cost carrier metadata."""
        if type(self.form) is not CommunicationMapForm:
            raise TypeError("map form must be CommunicationMapForm.")
        if type(self.transport_mode) is not CommunicationTransportMode:
            raise TypeError(
                "transport_mode must be CommunicationTransportMode."
            )
        if isinstance(self.edge_capacity, bool) or not isinstance(
            self.edge_capacity, Integral
        ):
            raise TypeError("edge_capacity must be an integral, not bool.")
        if self.edge_capacity < 0:
            raise ValueError("edge_capacity must be nonnegative.")


@dataclass(frozen=True, eq=False)
class PrescribedVolumeUpdate:
    """Retain optional caller-owned final per-box volumes by identity.

    ``final_volumes`` is ``None`` or later validates as a same-device contiguous
    pointer-backed ``wp.float64`` array shaped ``(B,)`` in m³.  P1 only
    preflights it; P2 owns any volume write.
    """

    final_volumes: object | None


@dataclass(frozen=True, eq=False)
class CommunicationConfiguration:
    """Combine immutable communication declarations without binding a session.

    The configuration preserves all nested carrier and Warp-array identities.
    It neither reads resident population nor a time step, so it cannot validate
    population-dependent outbound totals; P3 must do that before its writers.
    """

    communication_map: CommunicationMap
    prescribed_volume: PrescribedVolumeUpdate
    resource_shapes: tuple[CommunicationResourceShape, ...]

    def __post_init__(self) -> None:
        """Validate cheap nested carrier and unique-role metadata."""
        if type(self.communication_map) is not CommunicationMap:
            raise TypeError(
                "communication_map must be an exact CommunicationMap."
            )
        if type(self.prescribed_volume) is not PrescribedVolumeUpdate:
            raise TypeError(
                "prescribed_volume must be an exact PrescribedVolumeUpdate."
            )
        if type(self.resource_shapes) is not tuple:
            raise TypeError("resource_shapes must be an exact tuple.")
        if not all(
            type(item) is CommunicationResourceShape
            for item in self.resource_shapes
        ):
            raise TypeError(
                "resource_shapes must contain exact "
                "CommunicationResourceShape values."
            )
        roles = tuple(item.role for item in self.resource_shapes)
        if len(set(roles)) != len(roles):
            raise ValueError("resource_shapes roles must be unique.")


@wp.kernel
def _scan_enabled(
    enabled: wp.array(dtype=wp.int32), invalid: wp.array(dtype=wp.int32)
):
    index = wp.tid()
    value = enabled[index]
    if value != 0 and value != 1:
        wp.atomic_max(invalid, 0, 1)


@wp.kernel
def _scan_rates(
    rates: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)
):
    index = wp.tid()
    value = rates[index]
    if not wp.isfinite(value) or value < 0.0:
        wp.atomic_max(invalid, 0, 1)


@wp.kernel
def _scan_volumes(
    volumes: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)
):
    index = wp.tid()
    value = volumes[index]
    if not wp.isfinite(value) or value <= 0.0:
        wp.atomic_max(invalid, 0, 1)


@wp.kernel
def _scan_topology(
    source: wp.array(dtype=wp.int32),
    destination: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    boxes: int,
    one_dimensional: int,
    invalid: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    if enabled[index] == 1:
        left = source[index]
        right = destination[index]
        if (
            left < 0
            or left >= boxes
            or right < 0
            or right >= boxes
            or left == right
            or (one_dimensional == 1 and wp.abs(left - right) != 1)
        ):
            wp.atomic_max(invalid, 0, 1)


@wp.kernel
def _find_duplicates(
    source: wp.array(dtype=wp.int32),
    destination: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    boxes: int,
    table: wp.array(dtype=wp.int64),
    invalid: wp.array(dtype=wp.int32),
):
    """Insert enabled directed pairs into private hash storage."""
    index = wp.tid()
    if enabled[index] == 1:
        key = wp.int64(source[index]) * wp.int64(boxes) + wp.int64(
            destination[index]
        )
        size = table.shape[0]
        slot = int(key % wp.int64(size))
        for offset in range(size):
            candidate = (slot + offset) % size
            previous = wp.atomic_cas(table, candidate, wp.int64(-1), key)
            if previous == key:
                wp.atomic_max(invalid, 0, 1)
                break
            if previous == wp.int64(-1):
                break


def _is_warp_array(value: object) -> bool:
    """Return whether value is a concrete Warp array carrier."""
    return (
        type(value).__module__.startswith("warp")
        and type(value).__name__ == "array"
    )


def _array_range(
    value: Any, shape: tuple[int, ...], name: str, item_size: int
) -> tuple[int, int] | None:
    """Validate contiguous backing metadata and return a nonempty byte range."""
    expected: list[int] = []
    stride = item_size
    for length in reversed(shape):
        expected.insert(0, stride)
        stride *= length
    if getattr(value, "strides", None) != tuple(expected):
        raise ValueError(f"{name} must be contiguous.")
    count = 1
    for length in shape:
        count *= length
    if count == 0:
        return None
    pointer = getattr(value, "ptr", None)
    if not isinstance(pointer, Integral) or pointer <= 0:
        raise ValueError(f"{name} must have a valid pointer.")
    if pointer % item_size:
        raise ValueError(f"{name} pointer must be {item_size}-byte aligned.")
    capacity = getattr(value, "capacity", None)
    required = count * item_size
    if (
        isinstance(capacity, bool)
        or not isinstance(capacity, Integral)
        or capacity < required
        or capacity % item_size
    ):
        raise ValueError(
            f"{name} must have sufficient integral storage capacity."
        )
    return int(pointer), int(pointer) + required


def _validate_array(
    value: object,
    shape: tuple[int, ...],
    dtype: object,
    device: object,
    name: str,
    item_size: int,
) -> tuple[int, int] | None:
    """Validate one fixed-schema Warp array without reading its payload."""
    if not _is_warp_array(value):
        raise ValueError(f"{name} must be a Warp array.")
    array = cast(Any, value)
    if len(getattr(array, "shape", ())) != len(shape):
        raise ValueError(f"{name} must have rank {len(shape)}.")
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if array.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}.")
    if array.device != device:
        raise ValueError(f"{name} device must match device.")
    return _array_range(array, shape, name, item_size)


def _overlaps(
    left: tuple[int, int] | None, right: tuple[int, int] | None
) -> bool:
    """Return whether two nonempty half-open byte ranges overlap."""
    return (
        left is not None
        and right is not None
        and left[0] < right[1]
        and right[0] < left[1]
    )


def _scan(value: Any, size: int, kernel: Any, name: str, message: str) -> None:
    """Run one read-only scalar-status payload validation scan."""
    if size == 0:
        return
    device = cast(Any, value.device)
    invalid = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(kernel, dim=size, inputs=[value, invalid], device=device)
    if int(invalid.numpy()[0]) != 0:
        raise ValueError(f"{name} {message}.")


def _duplicate_table_size(edges: int) -> int:
    """Return private power-of-two hash capacity at no more than half load."""
    size = 1
    while size < max(2, 2 * edges):
        size *= 2
    return size


def _validate_resource_shapes(
    resource_shapes: object,
) -> tuple[CommunicationResourceShape, ...]:
    """Validate exact resource declarations before payload metadata."""
    if type(resource_shapes) is not tuple:
        raise TypeError("resource_shapes must be an exact tuple.")
    if not all(
        type(item) is CommunicationResourceShape for item in resource_shapes
    ):
        raise TypeError(
            "resource_shapes must contain exact "
            "CommunicationResourceShape values."
        )
    typed_shapes = cast(tuple[CommunicationResourceShape, ...], resource_shapes)
    for resource in typed_shapes:
        if type(resource.role) is not str:
            raise TypeError("resource role must be an exact str.")
        if not resource.role or resource.role != resource.role.strip():
            raise ValueError("resource role must be nonempty and unpadded.")
        if type(resource.shape_kind) is not CommunicationShapeKind:
            raise TypeError(
                "resource shape_kind must be CommunicationShapeKind."
            )
    roles = tuple(resource.role for resource in typed_shapes)
    if len(set(roles)) != len(roles):
        raise ValueError("resource_shapes roles must be unique.")
    return typed_shapes


def _validate_metadata(
    configuration: object, dimensions: object, device: object
) -> CommunicationConfiguration:
    """Validate exact outer carriers before inspecting any Warp metadata."""
    if type(configuration) is not CommunicationConfiguration:
        raise TypeError(
            "configuration must be an exact CommunicationConfiguration."
        )
    if type(dimensions) is not ResidentDimensions:
        raise TypeError("dimensions must be an exact ResidentDimensions.")
    if (
        device is None
        or not type(device).__module__.startswith("warp")
        or type(device).__name__ != "Device"
    ):
        raise TypeError("device must be non-null Warp device metadata.")
    typed = cast(CommunicationConfiguration, configuration)
    if type(typed.communication_map) is not CommunicationMap:
        raise TypeError("communication_map must be an exact CommunicationMap.")
    if type(typed.prescribed_volume) is not PrescribedVolumeUpdate:
        raise TypeError(
            "prescribed_volume must be an exact PrescribedVolumeUpdate."
        )
    _validate_resource_shapes(typed.resource_shapes)
    map_data = typed.communication_map
    if type(map_data.form) is not CommunicationMapForm:
        raise TypeError("map form must be CommunicationMapForm.")
    if type(map_data.transport_mode) is not CommunicationTransportMode:
        raise TypeError("transport_mode must be CommunicationTransportMode.")
    if isinstance(map_data.edge_capacity, bool) or not isinstance(
        map_data.edge_capacity, Integral
    ):
        raise TypeError("edge_capacity must be an integral, not bool.")
    if map_data.edge_capacity < 0:
        raise ValueError("edge_capacity must be nonnegative.")
    return typed


def validate_communication_configuration(
    configuration: object,
    dimensions: object,
    device: object,
) -> CommunicationConfiguration:
    """Read-only validate one fixed-shape communication configuration.

    Args:
        configuration: Exact immutable map, optional-volume, and shape carrier.
        dimensions: Exact resident dimensions; only ``n_boxes`` is used.
        device: Explicit active Warp device metadata for all payload arrays.

    Returns:
        The exact unchanged ``configuration`` object.

    Raises:
        TypeError: If outer carriers or device metadata are invalid.
        ValueError: If array schema, ownership, values, or enabled topology is
            invalid.

    Notes:
        This validates map topology and rate domains only.  It has neither
        source inventory nor time-step input and cannot validate outbound
        population totals; P3 must enforce that condition before writers run.
    """
    typed = _validate_metadata(configuration, dimensions, device)
    map_data = typed.communication_map
    volumes = typed.prescribed_volume.final_volumes
    boxes = cast(ResidentDimensions, dimensions).n_boxes
    warp_device = cast(Any, device)
    edges = int(map_data.edge_capacity)
    edge_shape = (edges,)

    source_range = _validate_array(
        map_data.source_boxes,
        edge_shape,
        wp.int32,
        warp_device,
        "source_boxes",
        4,
    )
    destination_range = _validate_array(
        map_data.destination_boxes,
        edge_shape,
        wp.int32,
        warp_device,
        "destination_boxes",
        4,
    )
    enabled_range = _validate_array(
        map_data.enabled,
        edge_shape,
        wp.int32,
        warp_device,
        "enabled",
        4,
    )
    rates_range = _validate_array(
        map_data.rates,
        edge_shape,
        wp.float64,
        warp_device,
        "rates",
        8,
    )
    volume_range = None
    if volumes is not None:
        volume_range = _validate_array(
            volumes,
            (boxes,),
            wp.float64,
            warp_device,
            "final_volumes",
            8,
        )

    values = (
        map_data.source_boxes,
        map_data.destination_boxes,
        map_data.enabled,
        map_data.rates,
        volumes,
    )
    ranges = (
        source_range,
        destination_range,
        enabled_range,
        rates_range,
        volume_range,
    )
    for index, value in enumerate(values):
        if value is None:
            continue
        for previous in range(index):
            if value is values[previous] or _overlaps(
                ranges[index], ranges[previous]
            ):
                raise ValueError("communication map arrays must not alias.")

    source = cast(Any, map_data.source_boxes)
    destination = cast(Any, map_data.destination_boxes)
    enabled = cast(Any, map_data.enabled)
    rates = cast(Any, map_data.rates)
    _scan(enabled, edges, _scan_enabled, "enabled", "values must be 0 or 1")
    _scan(
        rates,
        edges,
        _scan_rates,
        "rates",
        "values must be finite and nonnegative",
    )
    if volumes is not None:
        _scan(
            cast(Any, volumes),
            boxes,
            _scan_volumes,
            "final_volumes",
            "values must be finite and positive",
        )
    if edges == 0:
        return typed

    invalid = wp.zeros(1, dtype=wp.int32, device=warp_device)
    wp.launch(
        _scan_topology,
        dim=edges,
        inputs=[
            source,
            destination,
            enabled,
            boxes,
            int(map_data.form is CommunicationMapForm.ONE_DIMENSIONAL),
            invalid,
        ],
        device=warp_device,
    )
    if int(invalid.numpy()[0]) != 0:
        raise ValueError("enabled edges must have valid distinct topology.")

    table = wp.full(
        _duplicate_table_size(edges), -1, dtype=wp.int64, device=warp_device
    )
    invalid = wp.zeros(1, dtype=wp.int32, device=warp_device)
    wp.launch(
        _find_duplicates,
        dim=edges,
        inputs=[source, destination, enabled, boxes, table, invalid],
        device=warp_device,
    )
    if int(invalid.numpy()[0]) != 0:
        raise ValueError("enabled directed edge pairs must be unique.")
    return typed
