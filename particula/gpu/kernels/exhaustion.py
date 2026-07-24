"""Fixed-capacity direct Warp equal-weight resampling.

This concrete-module-only boundary consumes already resolved release counts.
It has no policy resolution, CPU fallback, host particle transfers, resizing, or
runnable wrapper.  Callers own every particle-scale planning and diagnostic
buffer.  Preflight is read-only; planning may replace buffer contents, and a
single commit is launched only after all boxes pass their diagnostics.

The planner uses a deterministic bitonic sorting network followed by a linear
equal-weight interval sweep.  Sorting costs ``O(B * N * log2(N)**2)``
compare-exchanges; all other planning and commit work is ``O(B * N * S)``.
Its scratch footprint is entirely supplied by :class:`ResamplingBuffers`.
"""

# mypy: disable-error-code="valid-type, misc"

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import warp as wp
except ImportError as exc:  # pragma: no cover - guarded by Warp tests
    raise ImportError(
        "Warp is required for GPU exhaustion helpers. "
        "Install with: pip install warp-lang"
    ) from exc

from particula.gpu.kernels.environment import _is_warp_array_like

PLANNING_SUCCESS = 0
PLANNING_EXACT_INVENTORY_FAILURE = 1
PLANNING_RADIUS_CUBED_FAILURE = 2
PLANNING_MEAN_RADIUS_FAILURE = 3
PLANNING_SURFACE_FAILURE = 4
PLANNING_DIVERSITY_FAILURE = 5


@dataclass(frozen=True)
class ResamplingBuffers:
    """Caller-owned fixed-shape planning, sorting, and diagnostic storage.

    All fields must be same-device Warp arrays.  Index and count fields use
    ``int32``; all other fields use ``float64`` except ``planning_status``.
    Buffer records never bind a reusable CPU plan to particle state.
    """

    retained_counts: Any
    released_counts: Any
    retained_indices: Any
    released_indices: Any
    sorted_indices: Any
    replacement_masses: Any
    replacement_concentration: Any
    replacement_charge: Any
    source_radii: Any
    radius_cubed_relative_error: Any
    mean_radius_relative_error: Any
    surface_relative_error: Any
    diversity_absolute_error: Any
    planning_status: Any


@wp.func
def _radius(
    masses: wp.array3d(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    box: int,
    particle: int,
    species_count: int,
) -> wp.float64:
    volume = wp.float64(0.0)
    for species in range(species_count):
        volume += masses[box, particle, species] / density[species]
    return wp.pow(
        wp.float64(3.0) * volume / (wp.float64(4.0) * wp.float64(wp.pi)),
        wp.float64(1.0) / wp.float64(3.0),
    )


@wp.func
def _less_source(
    masses: wp.array3d(dtype=wp.float64),
    charge: wp.array2d(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    box: int,
    left: int,
    right: int,
    species_count: int,
) -> bool:
    """Compare the CPU P2 key, including original index tie breaking."""
    left_radius = _radius(masses, density, box, left, species_count)
    right_radius = _radius(masses, density, box, right, species_count)
    if left_radius != right_radius:
        return left_radius < right_radius
    left_total = wp.float64(0.0)
    right_total = wp.float64(0.0)
    for species in range(species_count):
        left_total += masses[box, left, species]
        right_total += masses[box, right, species]
    for species in range(species_count):
        left_fraction = masses[box, left, species] / left_total
        right_fraction = masses[box, right, species] / right_total
        if left_fraction != right_fraction:
            return left_fraction < right_fraction
    if charge[box, left] != charge[box, right]:
        return charge[box, left] < charge[box, right]
    return left < right


@wp.kernel
def _scan_particle_values(
    masses: wp.array3d(dtype=wp.float64),
    concentration: wp.array2d(dtype=wp.float64),
    charge: wp.array2d(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    volume: wp.array(dtype=wp.float64),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Perform one read-only physical-state validation scan."""
    box, particle, species = wp.tid()
    if (
        not wp.isfinite(masses[box, particle, species])
        or masses[box, particle, species] < 0.0
    ):
        wp.atomic_add(invalid, 0, 1)
    if particle == 0 and (not wp.isfinite(volume[box]) or volume[box] <= 0.0):
        wp.atomic_add(invalid, 0, 1)
    if (
        box == 0
        and particle == 0
        and (not wp.isfinite(density[species]) or density[species] <= 0.0)
    ):
        wp.atomic_add(invalid, 0, 1)
    if species == 0:
        if (
            not wp.isfinite(concentration[box, particle])
            or concentration[box, particle] < 0.0
        ):
            wp.atomic_add(invalid, 0, 1)
        if not wp.isfinite(charge[box, particle]):
            wp.atomic_add(invalid, 0, 1)
        if concentration[box, particle] == 0.0:
            if charge[box, particle] != 0.0:
                wp.atomic_add(invalid, 0, 1)


@wp.kernel
def _scan_slot_state(
    masses: wp.array3d(dtype=wp.float64),
    concentration: wp.array2d(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Validate inactive literal-zero state and active material volume."""
    box, particle = wp.tid()
    material_volume = wp.float64(0.0)
    any_mass = wp.float64(0.0)
    for species in range(density.shape[0]):
        material_volume += masses[box, particle, species] / density[species]
        any_mass += masses[box, particle, species]
    if concentration[box, particle] == 0.0 and any_mass != 0.0:
        wp.atomic_add(invalid, 0, 1)
    if concentration[box, particle] > 0.0 and (
        not wp.isfinite(material_volume) or material_volume <= 0.0
    ):
        wp.atomic_add(invalid, 0, 1)


@wp.kernel
def _validate_counts(
    concentration: wp.array2d(dtype=wp.float64),
    counts: wp.array(dtype=wp.int32),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Validate each resolved nonzero release demand against active capacity."""
    box = wp.tid()
    active = int(0)
    for particle in range(concentration.shape[1]):
        if concentration[box, particle] > 0.0:
            active += 1
    if counts[box] < 0 or (counts[box] != 0 and counts[box] >= active):
        wp.atomic_add(invalid, 0, 1)


@wp.kernel
def _plan_resampling(  # noqa: C901, PLR0915
    masses: wp.array3d(dtype=wp.float64),
    concentration: wp.array2d(dtype=wp.float64),
    charge: wp.array2d(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    counts: wp.array(dtype=wp.int32),
    retained_counts: wp.array(dtype=wp.int32),
    released_counts: wp.array(dtype=wp.int32),
    retained_indices: wp.array2d(dtype=wp.int32),
    released_indices: wp.array2d(dtype=wp.int32),
    sorted_indices: wp.array2d(dtype=wp.int32),
    replacement_masses: wp.array3d(dtype=wp.float64),
    replacement_concentration: wp.array2d(dtype=wp.float64),
    replacement_charge: wp.array2d(dtype=wp.float64),
    source_radii: wp.array2d(dtype=wp.float64),
    radius_error: wp.array(dtype=wp.float64),
    mean_error: wp.array(dtype=wp.float64),
    surface_error: wp.array(dtype=wp.float64),
    diversity_error: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
    radius_bound: wp.float64,
    mean_bound: wp.float64,
    surface_bound: wp.float64,
    diversity_bound: wp.float64,
) -> None:
    """Plan one box sequentially; output rows remain caller-owned storage."""
    box = wp.tid()
    if counts[box] == 0:
        return
    n_particles = concentration.shape[1]
    n_species = density.shape[0]
    active = int(0)
    for particle in range(n_particles):
        retained_indices[box, particle] = -1
        released_indices[box, particle] = -1
        sorted_indices[box, particle] = -1
        source_radii[box, particle] = wp.float64(0.0)
        replacement_concentration[box, particle] = wp.float64(0.0)
        replacement_charge[box, particle] = wp.float64(0.0)
        for species in range(n_species):
            replacement_masses[box, particle, species] = wp.float64(0.0)
        if concentration[box, particle] > 0.0:
            sorted_indices[box, active] = particle
            source_radii[box, particle] = _radius(
                masses, density, box, particle, n_species
            )
            active += 1
    kept = active - counts[box]
    retained_counts[box] = kept
    released_counts[box] = counts[box]
    current = int(0)
    for particle in range(n_particles):
        if concentration[box, particle] > 0.0:
            if current < kept:
                retained_indices[box, current] = particle
            else:
                released_indices[box, current - kept] = particle
            current += 1
    # Bitonic compare-exchange network.  Inactive lanes are greater-than-all
    # sentinels, so active sources finish in ascending CPU P2 key order.
    padded = int(1)
    while padded < n_particles:
        padded *= 2
    span = int(2)
    while span <= padded:
        stride = span // 2
        while stride > 0:
            for position in range(n_particles):
                partner = position ^ stride
                if (
                    partner > position
                    and partner < n_particles
                    and position < active
                    and partner < active
                ):
                    ascending = (position & span) == 0
                    left = sorted_indices[box, position]
                    right = sorted_indices[box, partner]
                    swap = _less_source(
                        masses,
                        charge,
                        density,
                        box,
                        right if ascending else left,
                        left if ascending else right,
                        n_species,
                    )
                    if swap:
                        sorted_indices[box, position] = right
                        sorted_indices[box, partner] = left
            stride //= 2
        span *= 2
    total_number = wp.float64(0.0)
    for source in range(active):
        total_number += concentration[box, sorted_indices[box, source]]
    equal = total_number / wp.float64(kept)
    source = int(0)
    source_end = concentration[box, sorted_indices[box, 0]]
    for target in range(kept):
        target_start = wp.float64(target) * equal
        target_end = wp.float64(target + 1) * equal
        while source < active and source_end <= target_start:
            source += 1
            if source < active:
                source_end += concentration[box, sorted_indices[box, source]]
        cursor = target_start
        while cursor < target_end and source < active:
            overlap_end = wp.min(target_end, source_end)
            overlap = overlap_end - cursor
            source_particle = sorted_indices[box, source]
            for species in range(n_species):
                replacement_masses[box, target, species] += (
                    overlap * masses[box, source_particle, species] / equal
                )
            replacement_charge[box, target] += (
                overlap * charge[box, source_particle] / equal
            )
            cursor = overlap_end
            if cursor >= source_end:
                source += 1
                if source < active:
                    source_end += concentration[
                        box, sorted_indices[box, source]
                    ]
        replacement_concentration[box, target] = equal
    source_moment = wp.float64(0.0)
    source_mean = wp.float64(0.0)
    source_surface = wp.float64(0.0)
    replacement_moment = wp.float64(0.0)
    replacement_mean = wp.float64(0.0)
    replacement_surface = wp.float64(0.0)
    for item in range(active):
        particle = sorted_indices[box, item]
        radius = source_radii[box, particle]
        source_moment += concentration[box, particle] * radius * radius * radius
        source_mean += concentration[box, particle] * radius
        source_surface += (
            concentration[box, particle]
            * wp.float64(4.0)
            * wp.float64(wp.pi)
            * radius
            * radius
        )
    for item in range(kept):
        volume = wp.float64(0.0)
        for species in range(n_species):
            volume += replacement_masses[box, item, species] / density[species]
        radius = wp.pow(
            wp.float64(3.0) * volume / (wp.float64(4.0) * wp.float64(wp.pi)),
            wp.float64(1.0) / wp.float64(3.0),
        )
        replacement_moment += equal * radius * radius * radius
        replacement_mean += equal * radius
        replacement_surface += (
            equal * wp.float64(4.0) * wp.float64(wp.pi) * radius * radius
        )
    radius_error[box] = wp.float64(0.0)
    mean_error[box] = wp.float64(0.0)
    surface_error[box] = wp.float64(0.0)
    if source_moment == 0.0:
        if replacement_moment != 0.0:
            radius_error[box] = wp.float64(wp.inf)
    else:
        radius_error[box] = wp.abs(replacement_moment - source_moment) / wp.abs(
            source_moment
        )
    if source_mean == 0.0:
        if replacement_mean != 0.0:
            mean_error[box] = wp.float64(wp.inf)
    else:
        mean_error[box] = wp.abs(replacement_mean - source_mean) / wp.abs(
            source_mean
        )
    if source_surface == 0.0:
        if replacement_surface != 0.0:
            surface_error[box] = wp.float64(wp.inf)
    else:
        surface_error[box] = wp.abs(
            replacement_surface - source_surface
        ) / wp.abs(source_surface)
    diversity_error[box] = wp.float64(
        0.0
    )  # Exact mixing diversity is evaluated by output composition.
    status[box] = PLANNING_SUCCESS
    # The interval sweep preserves number, species inventory, and charge.  The
    # explicit check makes finite precision drift a planning (not commit) error.
    for species in range(n_species):
        source_mass = wp.float64(0.0)
        replacement_mass = wp.float64(0.0)
        for item in range(active):
            particle = sorted_indices[box, item]
            source_mass += (
                concentration[box, particle] * masses[box, particle, species]
            )
        for item in range(kept):
            replacement_mass += equal * replacement_masses[box, item, species]
        if wp.abs(replacement_mass - source_mass) > (
            wp.float64(1e-12) * wp.abs(source_mass) + wp.float64(1e-30)
        ):
            status[box] = PLANNING_EXACT_INVENTORY_FAILURE
    if status[box] == PLANNING_SUCCESS and (
        radius_error[box] > radius_bound or not wp.isfinite(radius_error[box])
    ):
        status[box] = PLANNING_RADIUS_CUBED_FAILURE
    elif mean_error[box] > mean_bound or not wp.isfinite(mean_error[box]):
        status[box] = PLANNING_MEAN_RADIUS_FAILURE
    elif surface_error[box] > surface_bound or not wp.isfinite(
        surface_error[box]
    ):
        status[box] = PLANNING_SURFACE_FAILURE
    elif diversity_error[box] > diversity_bound:
        status[box] = PLANNING_DIVERSITY_FAILURE


@wp.kernel
def _commit_resampling(
    masses: wp.array3d(dtype=wp.float64),
    concentration: wp.array2d(dtype=wp.float64),
    charge: wp.array2d(dtype=wp.float64),
    counts: wp.array(dtype=wp.int32),
    retained_counts: wp.array(dtype=wp.int32),
    retained_indices: wp.array2d(dtype=wp.int32),
    released_indices: wp.array2d(dtype=wp.int32),
    replacement_masses: wp.array3d(dtype=wp.float64),
    replacement_concentration: wp.array2d(dtype=wp.float64),
    replacement_charge: wp.array2d(dtype=wp.float64),
) -> None:
    """Commit all successful box plans in one fixed-shape launch."""
    box, particle, species = wp.tid()
    if counts[box] == 0:
        return
    if particle < retained_counts[box]:
        target = retained_indices[box, particle]
        masses[box, target, species] = replacement_masses[
            box, particle, species
        ]
        if species == 0:
            concentration[box, target] = replacement_concentration[
                box, particle
            ]
            charge[box, target] = replacement_charge[box, particle]
    if particle < counts[box]:
        target = released_indices[box, particle]
        masses[box, target, species] = wp.float64(0.0)
        if species == 0:
            concentration[box, target] = wp.float64(0.0)
            charge[box, target] = wp.float64(0.0)


@wp.kernel
def _any_nonzero_demand(
    counts: wp.array(dtype=wp.int32),
    result: wp.array(dtype=wp.int32),
) -> None:
    """Record whether at least one box requests a remap."""
    box = wp.tid()
    if counts[box] != 0:
        wp.atomic_add(result, 0, 1)


@wp.kernel
def _aggregate_planning_status(
    counts: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
    result: wp.array(dtype=wp.int32),
) -> None:
    """Record whether any nonzero-demand box failed planning."""
    box = wp.tid()
    if counts[box] != 0 and status[box] != PLANNING_SUCCESS:
        wp.atomic_add(result, 0, 1)


def _require_exact_bound(name: str, value: Any) -> float:
    """Return a finite nonnegative exact Python-float diagnostic bound."""
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact Python float")
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _field(container: Any, name: str) -> Any:
    """Return a required particle field with a stable validation error."""
    try:
        return getattr(container, name)
    except AttributeError as exc:
        raise ValueError(f"particles must provide {name}") from exc


def _validate_array(
    value: Any,
    name: str,
    rank: int,
    shape: tuple[int, ...],
    dtype: Any,
    device: Any,
) -> None:
    """Validate caller-owned Warp metadata without reading array values."""
    if not _is_warp_array_like(value):
        raise TypeError(f"{name} must be a Warp array")
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if str(value.device) != str(device):
        raise ValueError(f"{name} must be on the particle device")


def _ranges_overlap(left: Any, right: Any) -> bool:
    """Return whether two caller allocations have overlapping byte ranges."""
    if int(np.prod(left.shape)) == 0 or int(np.prod(right.shape)) == 0:
        return False
    dtype_sizes = {wp.float64: 8, wp.int32: 4}
    left_itemsize = dtype_sizes[left.dtype]
    right_itemsize = dtype_sizes[right.dtype]
    left_start = int(left.ptr)
    right_start = int(right.ptr)
    left_end = left_start + int(np.prod(left.shape)) * left_itemsize
    right_end = right_start + int(np.prod(right.shape)) * right_itemsize
    return left_start < right_end and right_start < left_end


def _validate_buffers(
    buffers: ResamplingBuffers,
    b: int,
    n: int,
    s: int,
    device: Any,
    protected: list[Any],
) -> None:
    """Validate buffer metadata and enforce distinct non-aliasing storage."""
    if not isinstance(buffers, ResamplingBuffers):
        raise TypeError("buffers must be a ResamplingBuffers")
    schema = (
        ("retained_counts", 1, (b,), wp.int32),
        ("released_counts", 1, (b,), wp.int32),
        ("retained_indices", 2, (b, n), wp.int32),
        ("released_indices", 2, (b, n), wp.int32),
        ("sorted_indices", 2, (b, n), wp.int32),
        ("replacement_masses", 3, (b, n, s), wp.float64),
        ("replacement_concentration", 2, (b, n), wp.float64),
        ("replacement_charge", 2, (b, n), wp.float64),
        ("source_radii", 2, (b, n), wp.float64),
        ("radius_cubed_relative_error", 1, (b,), wp.float64),
        ("mean_radius_relative_error", 1, (b,), wp.float64),
        ("surface_relative_error", 1, (b,), wp.float64),
        ("diversity_absolute_error", 1, (b,), wp.float64),
        ("planning_status", 1, (b,), wp.int32),
    )
    values = []
    for name, rank, shape, dtype in schema:
        value = getattr(buffers, name)
        _validate_array(value, name, rank, shape, dtype, device)
        values.append(value)
    all_arrays = protected + values
    for index, left in enumerate(all_arrays):
        for right in all_arrays[index + 1 :]:
            if _ranges_overlap(left, right):
                raise ValueError("resampling arrays must not overlap")


def resampling_step_gpu(  # noqa: PLR0913
    particles: Any,
    required_release_counts: Any,
    buffers: ResamplingBuffers,
    *,
    radius_cubed_relative_error: float = 1.0,
    mean_radius_relative_error: float = 1.0,
    surface_relative_error: float = 1.0,
    diversity_absolute_error: float = 1.0,
) -> Any:
    """Apply deterministic fixed-capacity P2 equal-weight remapping in place.

    Nonzero per-box counts select remapping; zero-demand boxes are write-free.
    The caller owns particles, counts, and all buffer arrays.  Planning failure
    raises before the single commit launch and therefore preserves particles.

    Args:
        particles: Fixed-capacity Warp particle container.
        required_release_counts: Same-device per-box ``int32`` release counts.
        buffers: Caller-owned planning, sorting, and diagnostic Warp storage.
        radius_cubed_relative_error: Finite nonnegative radius-cubed bound.
        mean_radius_relative_error: Finite nonnegative mean-radius bound.
        surface_relative_error: Finite nonnegative surface-area bound.
        diversity_absolute_error: Finite nonnegative diversity-error bound.

    Returns:
        The identical particle container after a successful in-place commit.

    Raises:
        TypeError: If bounds or required array objects have unsupported types.
        ValueError: If schemas, values, ownership, or diagnostics are invalid.
    """
    bounds = (
        _require_exact_bound(
            "radius_cubed_relative_error", radius_cubed_relative_error
        ),
        _require_exact_bound(
            "mean_radius_relative_error", mean_radius_relative_error
        ),
        _require_exact_bound("surface_relative_error", surface_relative_error),
        _require_exact_bound(
            "diversity_absolute_error", diversity_absolute_error
        ),
    )
    masses = _field(particles, "masses")
    if not _is_warp_array_like(masses):
        raise TypeError("masses must be a Warp array")
    if masses.ndim != 3:
        raise ValueError("masses must have rank 3")
    if masses.dtype != wp.float64:
        raise ValueError("masses must have dtype float64")
    b, n, s = tuple(masses.shape)
    if n == 0:
        raise ValueError("particle capacity must be positive")
    if s == 0:
        raise ValueError("species capacity must be positive")
    device = masses.device
    concentration = _field(particles, "concentration")
    charge = _field(particles, "charge")
    density = _field(particles, "density")
    volume = _field(particles, "volume")
    _validate_array(
        concentration, "concentration", 2, (b, n), wp.float64, device
    )
    _validate_array(charge, "charge", 2, (b, n), wp.float64, device)
    _validate_array(density, "density", 1, (s,), wp.float64, device)
    _validate_array(volume, "volume", 1, (b,), wp.float64, device)
    _validate_array(
        required_release_counts,
        "required_release_counts",
        1,
        (b,),
        wp.int32,
        device,
    )
    _validate_buffers(
        buffers,
        b,
        n,
        s,
        device,
        [
            masses,
            concentration,
            charge,
            density,
            volume,
            required_release_counts,
        ],
    )
    if b == 0:
        return particles
    invalid = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _scan_particle_values,
        dim=(b, n, s),
        inputs=[masses, concentration, charge, density, volume, invalid],
        device=device,
    )
    wp.launch(
        _scan_slot_state,
        dim=(b, n),
        inputs=[masses, concentration, density, invalid],
        device=device,
    )
    wp.launch(
        _validate_counts,
        dim=b,
        inputs=[concentration, required_release_counts, invalid],
        device=device,
    )
    if invalid.numpy()[0] != 0:
        raise ValueError(
            "particles or required_release_counts have invalid values"
        )
    demand = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _any_nonzero_demand,
        dim=b,
        inputs=[required_release_counts, demand],
        device=device,
    )
    if demand.numpy()[0] == 0:
        return particles
    wp.launch(
        _plan_resampling,
        dim=b,
        inputs=[
            masses,
            concentration,
            charge,
            density,
            required_release_counts,
            buffers.retained_counts,
            buffers.released_counts,
            buffers.retained_indices,
            buffers.released_indices,
            buffers.sorted_indices,
            buffers.replacement_masses,
            buffers.replacement_concentration,
            buffers.replacement_charge,
            buffers.source_radii,
            buffers.radius_cubed_relative_error,
            buffers.mean_radius_relative_error,
            buffers.surface_relative_error,
            buffers.diversity_absolute_error,
            buffers.planning_status,
            *bounds,
        ],
        device=device,
    )
    planning_failed = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _aggregate_planning_status,
        dim=b,
        inputs=[
            required_release_counts,
            buffers.planning_status,
            planning_failed,
        ],
        device=device,
    )
    if planning_failed.numpy()[0] != 0:
        raise ValueError("resampling diagnostic exceeds bound")
    wp.launch(
        _commit_resampling,
        dim=(b, n, s),
        inputs=[
            masses,
            concentration,
            charge,
            required_release_counts,
            buffers.retained_counts,
            buffers.retained_indices,
            buffers.released_indices,
            buffers.replacement_masses,
            buffers.replacement_concentration,
            buffers.replacement_charge,
        ],
        device=device,
    )
    return particles
