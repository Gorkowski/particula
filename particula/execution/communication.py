# mypy: disable-error-code="valid-type, misc, operator"

"""Declare and read-only validate fixed-capacity resident communication maps.

This direct-import-only P1 boundary retains caller-owned Warp arrays by
identity. It validates one-dimensional or arbitrary directed edge maps,
per-box strictly positive volumes (m³), and per-source outbound amount bounds.
Rates and bounds use the same operation-defined amount units. It never
transfers data, writes a map payload, registers resources, resizes or compacts
storage, or executes communication. Empty and all-disabled maps declare no
future write and are valid no-op declarations.

Validation precedence is deterministic: record and metadata carriers, array
schemas, storage aliasing, physical domains, enabled-edge topology, outbound
totals, then representation. All payload scans run on the declared Warp device;
only private scalar status is read back. Schema and alias checks are O(1) for
the fixed six carriers, domain and outbound checks are O(E + B), and duplicate
directed-edge detection is O(E²), where E is edge capacity and B is box count.
Validation uses private status and per-box total scratch only. It neither
mutates nor copies caller-owned records or arrays, so a rejected declaration
may be corrected and retried. Writer execution, transfer, and post-launch
rollback policy are deferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import warp as wp

from particula.execution import Backend, Device


class CommunicationMapForm(str, Enum):
    """Describe the fixed edge topology form."""

    ONE_DIMENSIONAL = "one_dimensional"
    PAIR = "pair"


class CommunicationTransportMode(str, Enum):
    """Describe the future transported resident quantity."""

    GAS = "gas"
    PARTICLE = "particle"


class CommunicationBoundaryMode(str, Enum):
    """Describe declared boundary semantics; P1 has no boundary writer."""

    CLOSED = "closed"
    OUTFLOW = "outflow"
    INFLOW = "inflow"


class CommunicationRepresentation(str, Enum):
    """Describe the sole P1-supported fixed particle representation."""

    PARTICLE_RESOLVED = "particle_resolved"


def _nonnegative_int(value: object, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")


@dataclass(frozen=True, eq=False)
class CommunicationDimensions:
    """Declare nonnegative fixed resident resource dimensions.

    Attributes:
        n_boxes: Number of resident boxes B.
        n_particles: Fixed particle capacity per box; not related to edge
            payloads during P1 validation.
        n_species: Fixed species capacity per box; not related to edge payloads
            during P1 validation.
    """

    n_boxes: int
    n_particles: int
    n_species: int

    def __post_init__(self) -> None:
        """Validate fixed nonnegative resource dimensions."""
        _nonnegative_int(self.n_boxes, "CommunicationDimensions.n_boxes")
        _nonnegative_int(
            self.n_particles, "CommunicationDimensions.n_particles"
        )
        _nonnegative_int(self.n_species, "CommunicationDimensions.n_species")


@dataclass(frozen=True, eq=False)
class CommunicationMap:
    """Retain caller-owned fixed-capacity communication edge lanes by identity.

    Attributes:
        form: One-dimensional neighboring-box or arbitrary directed-pair form.
        transport_mode: Future gas or particle transport selection; P1 does not
            transport either quantity.
        boundary_mode: Declared boundary behavior. All declared modes pass P1
            after prior checks because P1 has no boundary writer.
        representation: Fixed particle representation required by P1.
        source_indices: Contiguous same-device ``wp.int32`` edge sources with
            shape ``(E,)``.
        destination_indices: Contiguous same-device ``wp.int32`` edge
            destinations with shape ``(E,)``.
        enabled: Contiguous same-device ``wp.bool`` edge flags with shape
            ``(E,)``.
        rates: Contiguous same-device ``wp.float64`` requested outbound amounts
            with shape ``(E,)``. Amount units match outbound bounds.

    Array schemas, device placement, and values are intentionally deferred to
    :func:`validate_communication_declarations` to preserve validation
    precedence and avoid inspecting or copying payloads during construction.
    """

    form: CommunicationMapForm
    transport_mode: CommunicationTransportMode
    boundary_mode: CommunicationBoundaryMode
    representation: CommunicationRepresentation
    source_indices: Any
    destination_indices: Any
    enabled: Any
    rates: Any

    def __post_init__(self) -> None:
        """Validate only enum carriers without inspecting array payloads."""
        if type(self.form) is not CommunicationMapForm:
            raise TypeError(
                "CommunicationMap.form must be a CommunicationMapForm."
            )
        if type(self.transport_mode) is not CommunicationTransportMode:
            raise TypeError(
                "CommunicationMap.transport_mode must be a "
                "CommunicationTransportMode."
            )
        if type(self.boundary_mode) is not CommunicationBoundaryMode:
            raise TypeError(
                "CommunicationMap.boundary_mode must be a "
                "CommunicationBoundaryMode."
            )
        if type(self.representation) is not CommunicationRepresentation:
            raise TypeError(
                "CommunicationMap.representation must be a "
                "CommunicationRepresentation."
            )


@dataclass(frozen=True, eq=False)
class PrescribedVolumeDeclaration:
    """Retain caller-owned volume and outbound-bound declarations by identity.

    Attributes:
        volumes: Contiguous same-device ``wp.float64`` resident-box volumes in
            m³ with shape ``(B,)``. Validation requires finite positive values.
        outbound_bounds: Contiguous same-device ``wp.float64`` available
            transferable amounts per source box with shape ``(B,)``. Units
            match map rates; validation requires finite nonnegative values.
    """

    volumes: Any
    outbound_bounds: Any


@dataclass(frozen=True, eq=False)
class CommunicationResourceShapes:
    """Bind fixed dimensions and a declared Warp device for one map payload.

    Attributes:
        dimensions: Fixed resident dimensions defining B and retained capacity
            metadata.
        device: Declared configured ``Backend.WARP`` device for all six arrays.
    """

    dimensions: CommunicationDimensions
    device: Device

    def __post_init__(self) -> None:
        """Validate retained dimensions and device metadata carriers."""
        if type(self.dimensions) is not CommunicationDimensions:
            raise TypeError(
                "CommunicationResourceShapes.dimensions must be a "
                "CommunicationDimensions."
            )
        if type(self.device) is not Device:
            raise TypeError(
                "CommunicationResourceShapes.device must be a Device."
            )


@wp.kernel
def _domain_status(
    source: wp.array(dtype=wp.int32),
    destination: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.bool),
    rates: wp.array(dtype=wp.float64),
    volumes: wp.array(dtype=wp.float64),
    bounds: wp.array(dtype=wp.float64),
    boxes: int,
    status: wp.array(dtype=wp.int32),
) -> None:
    """Set a private status for invalid payload domains."""
    index = wp.tid()
    if index < source.shape[0]:
        if source[index] < 0 or source[index] >= boxes:
            wp.atomic_min(status, 0, 1)
        if destination[index] < 0 or destination[index] >= boxes:
            wp.atomic_min(status, 0, 2)
        if not wp.isfinite(rates[index]) or rates[index] < 0.0:
            wp.atomic_min(status, 0, 3)
    if index < volumes.shape[0]:
        if not wp.isfinite(volumes[index]) or volumes[index] <= 0.0:
            wp.atomic_min(status, 0, 4)
        if not wp.isfinite(bounds[index]) or bounds[index] < 0.0:
            wp.atomic_min(status, 0, 5)


@wp.kernel
def _topology_status(
    source: wp.array(dtype=wp.int32),
    destination: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.bool),
    one_dimensional: int,
    status: wp.array(dtype=wp.int32),
) -> None:
    """Set a private status for enabled-edge topology violations."""
    index = wp.tid()
    if index >= source.shape[0] or not enabled[index]:
        return
    if source[index] == destination[index]:
        wp.atomic_min(status, 0, 1)
    if one_dimensional != 0 and wp.abs(destination[index] - source[index]) != 1:
        wp.atomic_min(status, 0, 2)
    other = index + 1
    while other < source.shape[0]:
        if (
            enabled[other]
            and source[index] == source[other]
            and destination[index] == destination[other]
        ):
            wp.atomic_min(status, 0, 3)
        other += 1


@wp.kernel
def _sum_outbound(
    source: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.bool),
    rates: wp.array(dtype=wp.float64),
    totals: wp.array(dtype=wp.float64),
) -> None:
    """Accumulate enabled requested amounts into private per-source totals."""
    index = wp.tid()
    if index < source.shape[0] and enabled[index]:
        wp.atomic_add(totals, source[index], rates[index])


@wp.kernel
def _bound_status(
    totals: wp.array(dtype=wp.float64),
    bounds: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
) -> None:
    """Set a private status for overflow or source-bound violations."""
    index = wp.tid()
    if index < totals.shape[0] and (
        not wp.isfinite(totals[index]) or totals[index] > bounds[index]
    ):
        wp.atomic_min(status, 0, 1)


def _status(device: Any) -> Any:
    """Allocate the permitted private one-element validation status carrier."""
    return wp.full(1, 100, dtype=wp.int32, device=device)


def _dtype_itemsize(dtype: object) -> int:
    """Return the item size in bytes for the supported Warp scalar dtypes."""
    if dtype is wp.bool:
        return 1
    if dtype is wp.int32:
        return 4
    if dtype is wp.float64:
        return 8
    raise TypeError("Unsupported Warp dtype for alias validation.")


def _read_status(status: Any) -> int:
    """Read only private scalar validation status after its device scan."""
    return int(status.numpy()[0])


def _require_warp_array(
    value: Any, name: str, dtype: Any, shape: tuple[int]
) -> None:
    """Validate fixed Warp-array metadata without reading payload values."""
    required = ("shape", "dtype", "device", "ptr", "contiguous")
    if not all(hasattr(value, attribute) for attribute in required):
        raise TypeError(f"{name} must be a Warp array.")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}.")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if not value.contiguous:
        raise ValueError(f"{name} must be contiguous.")


def _validate_aliases(arrays: tuple[tuple[str, Any], ...]) -> None:
    """Reject overlapping nonempty Warp storage ranges from pointer metadata."""
    ranges: list[tuple[str, int, int]] = []
    for name, array in arrays:
        if array.shape[0] == 0:
            continue
        size = int(array.shape[0]) * _dtype_itemsize(array.dtype)
        start = int(array.ptr)
        ranges.append((name, start, start + size))
    for index, (name, start, end) in enumerate(ranges):
        for other_name, other_start, other_end in ranges[index + 1 :]:
            if start < other_end and other_start < end:
                raise ValueError(
                    f"{name} and {other_name} must not alias storage."
                )


def validate_communication_declarations(  # noqa: C901
    communication_map: CommunicationMap,
    prescribed_volume: PrescribedVolumeDeclaration,
    resource_shapes: CommunicationResourceShapes,
) -> tuple[
    CommunicationMap, PrescribedVolumeDeclaration, CommunicationResourceShapes
]:
    """Validate communication declarations without caller-visible mutation.

    Checks exact record/metadata carriers; six-array schemas, devices, and
    nonaliasing storage; physical domains; enabled-edge topology; strict
    per-source outbound bounds; and the supported representation, in that
    order. All lanes receive schema and domain validation, including disabled
    lanes. Enabled one-dimensional edges must connect distinct neighboring
    boxes; pair edges may connect any distinct boxes. Directed duplicates fail,
    while reverse edges are permitted. Enabled rates are summed per source and
    must not exceed its bound, with no tolerance.

    Valid empty, all-disabled, and zero-box declarations complete applicable
    checks and have no transfer effect. A nonempty edge table with zero boxes
    fails index validation. This read-only preflight uses device-side scans and
    private validation scratch, but does not transfer, copy, allocate, or
    mutate caller-owned payloads. It launches no communication writer; later
    phases own writer behavior and post-launch fault policy.

    Args:
        communication_map: Fixed-capacity directed edge declaration.
        prescribed_volume: Per-box volumes in m³ and source outbound bounds in
            the same amount units as map rates.
        resource_shapes: Fixed dimensions and declared Warp device.

    Returns:
        The exact supplied communication map, volume declaration, and resource
        shapes records, in that order.

    Raises:
        TypeError: If a required carrier, enum, scalar metadata value, or array
            dtype is invalid.
        ValueError: If the device, schema, aliasing, domain, topology, outbound
            total, or supported representation requirement is invalid.
    """
    if type(communication_map) is not CommunicationMap:
        raise TypeError("communication_map must be a CommunicationMap.")
    if type(prescribed_volume) is not PrescribedVolumeDeclaration:
        raise TypeError(
            "prescribed_volume must be a PrescribedVolumeDeclaration."
        )
    if type(resource_shapes) is not CommunicationResourceShapes:
        raise TypeError(
            "resource_shapes must be a CommunicationResourceShapes."
        )
    dimensions = resource_shapes.dimensions
    device = resource_shapes.device
    if type(dimensions) is not CommunicationDimensions:
        raise TypeError(
            "CommunicationResourceShapes.dimensions must be a "
            "CommunicationDimensions."
        )
    if type(device) is not Device:
        raise TypeError("CommunicationResourceShapes.device must be a Device.")
    if type(device.backend) is not Backend:
        raise TypeError(
            "CommunicationResourceShapes.device.backend must be a Backend."
        )
    if device.backend is not Backend.WARP:
        raise ValueError(
            "CommunicationResourceShapes.device must use Backend.WARP."
        )
    for value, name in (
        (dimensions.n_boxes, "n_boxes"),
        (dimensions.n_particles, "n_particles"),
        (dimensions.n_species, "n_species"),
    ):
        _nonnegative_int(value, f"CommunicationDimensions.{name}")
    if type(communication_map.form) is not CommunicationMapForm:
        raise TypeError("CommunicationMap.form has an invalid enum type.")
    if type(communication_map.transport_mode) is not CommunicationTransportMode:
        raise TypeError(
            "CommunicationMap.transport_mode has an invalid enum type."
        )
    if type(communication_map.boundary_mode) is not CommunicationBoundaryMode:
        raise TypeError(
            "CommunicationMap.boundary_mode has an invalid enum type."
        )
    if (
        type(communication_map.representation)
        is not CommunicationRepresentation
    ):
        raise TypeError(
            "CommunicationMap.representation has an invalid enum type."
        )
    try:
        warp_device = wp.get_device(device.native)
    except Exception as error:
        raise ValueError(
            "CommunicationResourceShapes.device must be a configured Warp "
            "device."
        ) from error
    arrays = (
        ("source_indices", communication_map.source_indices, wp.int32),
        (
            "destination_indices",
            communication_map.destination_indices,
            wp.int32,
        ),
        ("enabled", communication_map.enabled, wp.bool),
        ("rates", communication_map.rates, wp.float64),
        ("volumes", prescribed_volume.volumes, wp.float64),
        ("outbound_bounds", prescribed_volume.outbound_bounds, wp.float64),
    )
    source_shape = getattr(communication_map.source_indices, "shape", None)
    if not isinstance(source_shape, tuple):
        raise TypeError("source_indices must be a Warp array.")
    if len(source_shape) != 1:
        raise ValueError("source_indices must have rank 1.")
    edge_count = source_shape[0]
    if isinstance(edge_count, bool) or not isinstance(edge_count, int):
        raise TypeError("source_indices must have an integer shape.")
    for name, array, dtype in arrays[:4]:
        _require_warp_array(array, name, dtype, (edge_count,))
        if array.device != warp_device:
            raise ValueError(
                f"{name} must use CommunicationResourceShapes.device."
            )
    for name, array, dtype in arrays[4:]:
        _require_warp_array(array, name, dtype, (dimensions.n_boxes,))
        if array.device != warp_device:
            raise ValueError(
                f"{name} must use CommunicationResourceShapes.device."
            )
    _validate_aliases(tuple((name, array) for name, array, _ in arrays))
    status = _status(warp_device)
    wp.launch(
        _domain_status,
        dim=max(edge_count, dimensions.n_boxes),
        inputs=[
            communication_map.source_indices,
            communication_map.destination_indices,
            communication_map.enabled,
            communication_map.rates,
            prescribed_volume.volumes,
            prescribed_volume.outbound_bounds,
            dimensions.n_boxes,
            status,
        ],
        device=warp_device,
    )
    code = _read_status(status)
    messages = {
        1: "source_indices contain an out-of-range index.",
        2: "destination_indices contain an out-of-range index.",
        3: "rates must be finite and nonnegative.",
        4: "volumes must be finite and strictly positive.",
        5: "outbound_bounds must be finite and nonnegative.",
    }
    if code != 100:
        raise ValueError(messages[code])
    status = _status(warp_device)
    wp.launch(
        _topology_status,
        dim=edge_count,
        inputs=[
            communication_map.source_indices,
            communication_map.destination_indices,
            communication_map.enabled,
            int(communication_map.form is CommunicationMapForm.ONE_DIMENSIONAL),
            status,
        ],
        device=warp_device,
    )
    code = _read_status(status)
    messages = {
        1: "Enabled edges must not be self edges.",
        2: "ONE_DIMENSIONAL enabled edges must connect neighboring boxes.",
        3: "Enabled directed edges must be unique.",
    }
    if code != 100:
        raise ValueError(messages[code])
    totals = wp.zeros(dimensions.n_boxes, dtype=wp.float64, device=warp_device)
    wp.launch(
        _sum_outbound,
        dim=edge_count,
        inputs=[
            communication_map.source_indices,
            communication_map.enabled,
            communication_map.rates,
            totals,
        ],
        device=warp_device,
    )
    status = _status(warp_device)
    wp.launch(
        _bound_status,
        dim=dimensions.n_boxes,
        inputs=[totals, prescribed_volume.outbound_bounds, status],
        device=warp_device,
    )
    if _read_status(status) != 100:
        raise ValueError(
            "Enabled outbound totals must be finite and not exceed "
            "outbound_bounds."
        )
    if (
        communication_map.representation
        is not CommunicationRepresentation.PARTICLE_RESOLVED
    ):
        raise ValueError(
            "CommunicationMap.representation must be PARTICLE_RESOLVED."
        )
    return communication_map, prescribed_volume, resource_shapes
