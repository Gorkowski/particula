"""Focused contract checks for direct fixed-capacity GPU resampling."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

pytestmark = [pytest.mark.warp, pytest.mark.gpu_parity]


def _warp():
    """Import Warp only while executing a Warp-targeted test."""
    return pytest.importorskip("warp")


def _state(device: str = "cpu"):
    """Create a sparse two-species fixed-capacity particle state and buffers."""
    wp = _warp()
    from particula.gpu.kernels.exhaustion import ResamplingBuffers

    particles = SimpleNamespace(
        masses=wp.array(
            np.array([[[1.0, 0.0], [2.0, 1.0], [4.0, 0.0], [0.0, 0.0]]]),
            dtype=wp.float64,
            device=device,
        ),
        concentration=wp.array(
            [[2.0, 3.0, 5.0, 0.0]], dtype=wp.float64, device=device
        ),
        charge=wp.array(
            [[1.0, 2.0, 3.0, 0.0]], dtype=wp.float64, device=device
        ),
        density=wp.array([1.0, 1.0], dtype=wp.float64, device=device),
        volume=wp.array([1.0], dtype=wp.float64, device=device),
    )
    buffers = ResamplingBuffers(
        retained_counts=wp.zeros(1, dtype=wp.int32, device=device),
        released_counts=wp.zeros(1, dtype=wp.int32, device=device),
        retained_indices=wp.zeros((1, 4), dtype=wp.int32, device=device),
        released_indices=wp.zeros((1, 4), dtype=wp.int32, device=device),
        sorted_indices=wp.zeros((1, 4), dtype=wp.int32, device=device),
        replacement_masses=wp.zeros((1, 4, 2), dtype=wp.float64, device=device),
        replacement_concentration=wp.zeros(
            (1, 4), dtype=wp.float64, device=device
        ),
        replacement_charge=wp.zeros((1, 4), dtype=wp.float64, device=device),
        source_radii=wp.zeros((1, 4), dtype=wp.float64, device=device),
        radius_cubed_relative_error=wp.zeros(
            1, dtype=wp.float64, device=device
        ),
        mean_radius_relative_error=wp.zeros(1, dtype=wp.float64, device=device),
        surface_relative_error=wp.zeros(1, dtype=wp.float64, device=device),
        diversity_absolute_error=wp.zeros(1, dtype=wp.float64, device=device),
        planning_status=wp.zeros(1, dtype=wp.int32, device=device),
    )
    counts = wp.array([1], dtype=wp.int32, device=device)
    return particles, counts, buffers


def _custom_state(
    masses: np.ndarray,
    concentration: np.ndarray,
    charge: np.ndarray,
    density: np.ndarray,
    counts: np.ndarray,
    device: str = "cpu",
):
    """Create a custom Warp state for targeted parity and validation cases."""
    wp = _warp()
    from particula.gpu.kernels.exhaustion import ResamplingBuffers

    box_count, particle_count, species_count = masses.shape
    particles = SimpleNamespace(
        masses=wp.array(masses, dtype=wp.float64, device=device),
        concentration=wp.array(concentration, dtype=wp.float64, device=device),
        charge=wp.array(charge, dtype=wp.float64, device=device),
        density=wp.array(density, dtype=wp.float64, device=device),
        volume=wp.array(np.ones(box_count), dtype=wp.float64, device=device),
    )
    buffers = ResamplingBuffers(
        retained_counts=wp.zeros(box_count, dtype=wp.int32, device=device),
        released_counts=wp.zeros(box_count, dtype=wp.int32, device=device),
        retained_indices=wp.zeros(
            (box_count, particle_count), dtype=wp.int32, device=device
        ),
        released_indices=wp.zeros(
            (box_count, particle_count), dtype=wp.int32, device=device
        ),
        sorted_indices=wp.zeros(
            (box_count, particle_count), dtype=wp.int32, device=device
        ),
        replacement_masses=wp.zeros(
            (box_count, particle_count, species_count),
            dtype=wp.float64,
            device=device,
        ),
        replacement_concentration=wp.zeros(
            (box_count, particle_count), dtype=wp.float64, device=device
        ),
        replacement_charge=wp.zeros(
            (box_count, particle_count), dtype=wp.float64, device=device
        ),
        source_radii=wp.zeros(
            (box_count, particle_count), dtype=wp.float64, device=device
        ),
        radius_cubed_relative_error=wp.zeros(
            box_count, dtype=wp.float64, device=device
        ),
        mean_radius_relative_error=wp.zeros(
            box_count, dtype=wp.float64, device=device
        ),
        surface_relative_error=wp.zeros(
            box_count, dtype=wp.float64, device=device
        ),
        diversity_absolute_error=wp.zeros(
            box_count, dtype=wp.float64, device=device
        ),
        planning_status=wp.zeros(box_count, dtype=wp.int32, device=device),
    )
    return particles, wp.array(counts, dtype=wp.int32, device=device), buffers


def _single_active_state():
    """Create a valid state with one active slot and reusable buffers."""
    wp = _warp()
    particles, _, buffers = _state()
    particles.masses = wp.array(
        np.array([[[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]),
        dtype=wp.float64,
        device="cpu",
    )
    particles.concentration = wp.array(
        [[2.0, 0.0, 0.0, 0.0]], dtype=wp.float64, device="cpu"
    )
    particles.charge = wp.zeros((1, 4), dtype=wp.float64, device="cpu")
    return particles, buffers


def _snapshot(*containers: object) -> list[np.ndarray]:
    """Copy all Warp fields in containers for exact non-mutation checks."""
    return [
        value.numpy().copy()
        for container in containers
        for value in vars(container).values()
    ]


def _assert_snapshot(snapshot: list[np.ndarray], *containers: object) -> None:
    """Assert all Warp fields in containers retain their copied values."""
    values = [
        value for container in containers for value in vars(container).values()
    ]
    for value, expected in zip(values, snapshot, strict=True):
        npt.assert_array_equal(value.numpy(), expected)


def test_resampling_remaps_retained_original_slots_and_conserves_inventory() -> (
    None
):
    """A valid P2 call commits equal rows and clears released original slots."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    particles, counts, buffers = _state()
    before_mass = np.sum(
        particles.masses.numpy() * particles.concentration.numpy()[..., None],
        axis=1,
    )
    before_charge = np.sum(
        particles.charge.numpy() * particles.concentration.numpy(), axis=1
    )
    assert resampling_step_gpu(particles, counts, buffers) is particles
    assert buffers.retained_indices.numpy()[0, :2].tolist() == [0, 1]
    assert buffers.released_indices.numpy()[0, :1].tolist() == [2]
    assert buffers.retained_indices.numpy()[0, 2:].tolist() == [-1, -1]
    assert particles.concentration.numpy()[0, 2] == 0.0
    assert np.all(particles.masses.numpy()[0, 2] == 0.0)
    after_mass = np.sum(
        particles.masses.numpy() * particles.concentration.numpy()[..., None],
        axis=1,
    )
    after_charge = np.sum(
        particles.charge.numpy() * particles.concentration.numpy(), axis=1
    )
    npt.assert_allclose(after_mass, before_mass, rtol=1e-12, atol=1e-30)
    npt.assert_allclose(after_charge, before_charge, rtol=1e-12, atol=1e-30)


def test_resampling_zero_demand_is_write_free() -> None:
    """All-zero demand preserves particles and every caller-owned buffer."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    particles, counts, buffers = _state()
    counts = _warp().zeros(1, dtype=_warp().int32, device="cpu")
    count_container = SimpleNamespace(counts=counts)
    snapshots = _snapshot(particles, count_container, buffers)
    assert resampling_step_gpu(particles, counts, buffers) is particles
    _assert_snapshot(snapshots, particles, count_container, buffers)


def test_resampling_empty_box_with_zero_demand_is_write_free() -> None:
    """An empty valid box and zero demand leave all caller storage untouched."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    particles, buffers = _single_active_state()
    wp = _warp()
    particles.masses = wp.zeros((1, 4, 2), dtype=wp.float64, device="cpu")
    particles.concentration = wp.zeros((1, 4), dtype=wp.float64, device="cpu")
    counts = wp.zeros(1, dtype=wp.int32, device="cpu")
    count_container = SimpleNamespace(counts=counts)
    snapshots = _snapshot(particles, count_container, buffers)

    assert resampling_step_gpu(particles, counts, buffers) is particles

    _assert_snapshot(snapshots, particles, count_container, buffers)


def test_resampling_accepts_maximum_valid_release_count() -> None:
    """Releasing active-count minus one leaves exactly one retained slot."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    particles, _, buffers = _state()
    wp = _warp()
    counts = wp.array([2], dtype=wp.int32, device="cpu")

    assert resampling_step_gpu(particles, counts, buffers) is particles
    assert buffers.retained_counts.numpy().tolist() == [1]
    assert buffers.released_counts.numpy().tolist() == [2]
    assert np.count_nonzero(particles.concentration.numpy()) == 1


def test_resampling_rejects_release_from_single_active_slot_without_mutation() -> (
    None
):
    """A nonzero demand cannot release the only active slot."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    particles, buffers = _single_active_state()
    wp = _warp()
    counts = wp.array([1], dtype=wp.int32, device="cpu")
    count_container = SimpleNamespace(counts=counts)
    snapshots = _snapshot(particles, count_container, buffers)

    with pytest.raises(
        ValueError,
        match="particles or required_release_counts have invalid values",
    ):
        resampling_step_gpu(particles, counts, buffers)

    _assert_snapshot(snapshots, particles, count_container, buffers)


def test_resampling_failed_diagnostics_do_not_commit_particles() -> None:
    """A planning diagnostic failure leaves the caller particle state intact."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    particles, counts, buffers = _state()
    snapshots = [value.numpy().copy() for value in vars(particles).values()]

    with pytest.raises(ValueError, match="resampling diagnostic exceeds bound"):
        resampling_step_gpu(
            particles,
            counts,
            buffers,
            radius_cubed_relative_error=0.0,
            mean_radius_relative_error=0.0,
            surface_relative_error=0.0,
            diversity_absolute_error=0.0,
        )

    for value, expected in zip(
        vars(particles).values(), snapshots, strict=True
    ):
        npt.assert_array_equal(value.numpy(), expected)


def test_resampling_orders_sources_by_cpu_key_stably() -> None:
    """The direct planner sorts active sources by the documented CPU key."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    masses = np.array([[[5.0], [1.0], [3.0], [2.0], [4.0]]], dtype=np.float64)
    concentration = np.ones((1, 5), dtype=np.float64)
    charge = np.zeros((1, 5), dtype=np.float64)
    density = np.array([1.0], dtype=np.float64)
    counts = np.array([2], dtype=np.int32)
    particles, counts, buffers = _custom_state(
        masses, concentration, charge, density, counts
    )

    assert resampling_step_gpu(particles, counts, buffers) is particles
    assert buffers.sorted_indices.numpy()[0].tolist() == [1, 3, 2, 4, 0]
    assert buffers.retained_indices.numpy()[0, :3].tolist() == [0, 1, 2]
    assert buffers.released_indices.numpy()[0, :2].tolist() == [3, 4]


def test_resampling_rejects_invalid_density_without_mutation() -> None:
    """Density validation rejects nonpositive species values before planning."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    particles, counts, buffers = _state()
    wp = _warp()
    particles.density = wp.array([1.0, 0.0], dtype=wp.float64, device="cpu")
    count_container = SimpleNamespace(counts=counts)
    snapshots = _snapshot(particles, count_container, buffers)

    with pytest.raises(
        ValueError,
        match="particles or required_release_counts have invalid values",
    ):
        resampling_step_gpu(particles, counts, buffers)

    _assert_snapshot(snapshots, particles, count_container, buffers)


def test_resampling_rejects_diversity_bound_failure_without_commit() -> None:
    """The planning pass surfaces diversity violations before commit."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    particles, counts, buffers = _custom_state(
        masses=np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float64),
        concentration=np.array([[1.0, 1.0]], dtype=np.float64),
        charge=np.zeros((1, 2), dtype=np.float64),
        density=np.array([1.0, 1.0], dtype=np.float64),
        counts=np.array([1], dtype=np.int32),
    )
    count_container = SimpleNamespace(counts=counts)
    snapshots = _snapshot(particles, count_container)

    with pytest.raises(ValueError, match="resampling diagnostic exceeds bound"):
        resampling_step_gpu(
            particles,
            counts,
            buffers,
            diversity_absolute_error=0.0,
        )

    _assert_snapshot(snapshots, particles, count_container)
    assert buffers.planning_status.numpy().tolist() == [5]


@pytest.mark.parametrize("count", [-1, 3])
def test_resampling_rejects_invalid_release_counts_without_mutation(
    count: int,
) -> None:
    """Release counts outside the active-capacity range are preflight errors."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    particles, _, buffers = _state()
    wp = _warp()
    counts = wp.array([count], dtype=wp.int32, device="cpu")
    snapshots = [value.numpy().copy() for value in vars(particles).values()]
    snapshots.extend(value.numpy().copy() for value in vars(buffers).values())

    with pytest.raises(
        ValueError,
        match="particles or required_release_counts have invalid values",
    ):
        resampling_step_gpu(particles, counts, buffers)

    values = list(vars(particles).values()) + list(vars(buffers).values())
    for value, expected in zip(values, snapshots, strict=True):
        npt.assert_array_equal(value.numpy(), expected)


@pytest.mark.parametrize("bound", [0, np.float64(0.0), -1.0, float("nan")])
def test_resampling_rejects_non_exact_or_invalid_bounds(bound: Any) -> None:
    """Diagnostic bounds accept only finite nonnegative exact Python floats."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    particles, counts, buffers = _state()
    count_container = SimpleNamespace(counts=counts)
    snapshots = _snapshot(particles, count_container, buffers)
    with pytest.raises(
        (TypeError, ValueError), match="radius_cubed_relative_error"
    ):
        resampling_step_gpu(  # type: ignore[arg-type]
            particles, counts, buffers, radius_cubed_relative_error=bound
        )
    _assert_snapshot(snapshots, particles, count_container, buffers)


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.performance
def test_resampling_benchmark_smoke_uses_opt_in_marker() -> None:
    """Keep allocation-sensitive timing evidence behind the benchmark opt-in."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    particles, counts, buffers = _state()
    assert resampling_step_gpu(particles, counts, buffers) is particles
