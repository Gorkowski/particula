"""Contract tests for read-only particle-capacity exhaustion planning."""

from dataclasses import FrozenInstanceError

import numpy as np
import numpy.testing as npt
import pytest
from particula.particles.exhaustion import (
    POLICY_ACTIVATE,
    POLICY_RESAMPLE_DEFERRED,
    POLICY_SCALE_DEFERRED,
    ExhaustionBoxPlan,
    ExhaustionControls,
    ExhaustionInputs,
    ExhaustionPlan,
    WeightedInventory,
    _ResamplingBoxPlan,
    _ResamplingPlan,
    apply_resampling,
    get_weighted_inventory,
    plan_resampling,
    resolve_exhaustion,
)
from particula.particles.particle_data import ParticleData


def _inputs(
    requested: list[int] | None = None,
    free: list[int] | None = None,
    releasable: list[int] | None = None,
    indices: list[list[int]] | None = None,
) -> ExhaustionInputs:
    """Create deterministic int32 sidecars."""
    return ExhaustionInputs(
        requested_count=np.array(
            [2] if requested is None else requested,
            dtype=np.int32,
        ),
        free_count=np.array([3] if free is None else free, dtype=np.int32),
        resampling_releasable_count=np.array(
            [0] if releasable is None else releasable,
            dtype=np.int32,
        ),
        free_indices=np.array(
            [[0, 1, 3, -1]] if indices is None else indices,
            dtype=np.int32,
        ),
    )


def _sidecar_snapshots(inputs: ExhaustionInputs) -> tuple[bytes, ...]:
    """Return byte snapshots proving resolver read-only behavior."""
    return tuple(
        values.tobytes()
        for values in (
            inputs.requested_count,
            inputs.free_count,
            inputs.resampling_releasable_count,
            inputs.free_indices,
        )
    )


def test_strict_controls_defaults_and_frozen_records() -> None:
    """Public controls and records are strict, frozen, and tuple-backed."""
    controls = ExhaustionControls()
    assert controls.resampling is True
    assert controls.representative_volume_scaling is False
    with pytest.raises(FrozenInstanceError):
        controls.resampling = False  # type: ignore[misc]
    for value in (0, 1, np.bool_(True)):
        with pytest.raises(TypeError, match="exact Python bool"):
            ExhaustionControls(resampling=value)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="exact Python bool"):
            ExhaustionControls(
                representative_volume_scaling=value  # type: ignore[arg-type]
            )

    plan = resolve_exhaustion(_inputs(), controls)
    assert isinstance(plan, ExhaustionPlan)
    assert isinstance(plan.box_plans, tuple)
    assert isinstance(plan.box_plans[0], ExhaustionBoxPlan)
    with pytest.raises(FrozenInstanceError):
        plan.box_plans[0].admitted_count = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        _inputs().free_count = np.empty(0, dtype=np.int32)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("controls", "requested", "free", "releasable", "policy"),
    [
        (ExhaustionControls(), 2, 3, 0, POLICY_ACTIVATE),
        (ExhaustionControls(False, False), 2, 3, 0, POLICY_ACTIVATE),
        (ExhaustionControls(True, True), 4, 2, 2, POLICY_RESAMPLE_DEFERRED),
        (ExhaustionControls(False, True), 4, 2, 0, POLICY_SCALE_DEFERRED),
        (ExhaustionControls(True, False), 4, 2, 2, POLICY_RESAMPLE_DEFERRED),
        (ExhaustionControls(True, True), 4, 2, 1, POLICY_SCALE_DEFERRED),
    ],
)
def test_policy_precedence_and_exact_diagnostics(
    controls: ExhaustionControls,
    requested: int,
    free: int,
    releasable: int,
    policy: int,
) -> None:
    """Activation is flag-independent and resampling precedes scaling."""
    indices = [[0, 2, 3, -1]] if free >= 3 else [[0, 2, -1, -1]]
    inputs = _inputs([requested], [free], [releasable], indices)
    box_plan = resolve_exhaustion(inputs, controls).box_plans[0]
    assert box_plan.requested_count == requested
    assert box_plan.admitted_count == requested
    assert box_plan.required_release_count == max(requested - free, 0)
    assert box_plan.releasable_count == releasable
    assert box_plan.policy_code == policy
    assert box_plan.release_indices == ()
    assert np.isnan(box_plan.scale_factor)
    expected_indices = (0, 2) if policy == POLICY_ACTIVATE else ()
    assert box_plan.activation_indices == expected_indices


def test_both_disabled_exhaustion_fails_without_mutating_inputs() -> None:
    """Unrepresentable capacity fails closed without partial plans or writes."""
    inputs = _inputs([4], [2], [2], [[0, 2, -1, -1]])
    snapshot = _sidecar_snapshots(inputs)
    with pytest.raises(
        ValueError,
        match="exhaustion policy cannot represent requested capacity",
    ):
        resolve_exhaustion(inputs, ExhaustionControls(False, False))
    assert _sidecar_snapshots(inputs) == snapshot


def test_insufficient_resampling_without_scaling_fails_without_writes() -> None:
    """Insufficient resampling fails closed when scaling is disabled."""
    inputs = _inputs([4], [2], [1], [[0, 2, -1, -1]])
    snapshot = _sidecar_snapshots(inputs)
    with pytest.raises(
        ValueError,
        match="exhaustion policy cannot represent requested capacity",
    ):
        resolve_exhaustion(inputs, ExhaustionControls(True, False))
    assert _sidecar_snapshots(inputs) == snapshot


def test_later_invalid_box_blocks_earlier_resolution_and_mutation() -> None:
    """All boxes validate before a successful plan can be returned."""
    inputs = _inputs([1, 1], [2, 2], [0, 0], [[0, 1, -1], [2, 1, -1]])
    snapshot = _sidecar_snapshots(inputs)
    with pytest.raises(ValueError, match="strictly ascending"):
        resolve_exhaustion(inputs, ExhaustionControls())
    assert _sidecar_snapshots(inputs) == snapshot


@pytest.mark.parametrize(
    ("field", "value", "exception", "message"),
    [
        ("requested_count", [1], TypeError, "numpy array"),
        ("free_count", np.array([1], dtype=np.int64), TypeError, "dtype int32"),
        ("releasable", np.array([[1]], dtype=np.int32), ValueError, "rank 1"),
        ("indices", np.array([1], dtype=np.int32), ValueError, "rank 2"),
    ],
)
def test_sidecar_schema_validation(
    field: str,
    value: object,
    exception: type[Exception],
    message: str,
) -> None:
    """Resolver rejects non-coercible sidecar schemas."""
    inputs = _inputs()
    arguments = dict(inputs.__dict__)
    field_name = (
        "resampling_releasable_count" if field == "releasable" else field
    )
    if field == "indices":
        field_name = "free_indices"
    arguments[field_name] = value
    with pytest.raises(exception, match=message):
        resolve_exhaustion(ExhaustionInputs(**arguments), ExhaustionControls())


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        (_inputs([1], [1, 2], [0], [[0, -1]]), "matching box"),
        (
            _inputs([-1], [1], [0], [[0, -1]]),
            "requested_count must be nonnegative",
        ),
        (_inputs([1], [3], [0], [[0, -1]]), "free_count exceeds"),
        (_inputs([3], [1], [0], [[0, -1]]), "requested_count exceeds"),
        (_inputs([1], [1], [3], [[0, -1]]), "releasable_count exceeds"),
        (_inputs([1], [1], [0], [[2, -1]]), "out-of-range"),
        (_inputs([1], [2], [0], [[1, 0]]), "strictly ascending"),
        (_inputs([1], [1], [0], [[0, 1]]), "unused suffix"),
    ],
)
def test_sidecar_value_validation(
    inputs: ExhaustionInputs, message: str
) -> None:
    """Resolver enforces capacities, prefix ordering, ranges, and sentinels."""
    with pytest.raises(ValueError, match=message):
        resolve_exhaustion(inputs, ExhaustionControls())


def test_resolver_type_and_empty_capacity_cases() -> None:
    """Wrong public objects and valid empty dimensions have defined behavior."""
    with pytest.raises(TypeError, match="inputs must"):
        resolve_exhaustion(object(), ExhaustionControls())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="controls must"):
        resolve_exhaustion(_inputs(), object())  # type: ignore[arg-type]

    zero_boxes = ExhaustionInputs(
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty((0, 0), dtype=np.int32),
    )
    assert resolve_exhaustion(zero_boxes, ExhaustionControls()).box_plans == ()
    zero_capacity = _inputs([0], [0], [0], [[]])
    box_plan = resolve_exhaustion(
        zero_capacity, ExhaustionControls()
    ).box_plans[0]
    assert box_plan.policy_code == POLICY_ACTIVATE
    assert box_plan.activation_indices == ()
    exhausted_zero_capacity = _inputs([1], [0], [0], [[]])
    with pytest.raises(ValueError, match="requested_count exceeds"):
        resolve_exhaustion(
            exhausted_zero_capacity, ExhaustionControls(False, False)
        )


def test_activation_tuple_is_independent_of_sidecar_after_resolution() -> None:
    """Returned activation indices cannot alias editable caller sidecars."""
    inputs = _inputs([2], [3], [0], [[0, 1, 3, -1]])
    plan = resolve_exhaustion(inputs, ExhaustionControls())
    inputs.free_indices[0, 0] = 3
    assert plan.box_plans[0].activation_indices == (0, 1)


def test_weighted_inventory_matches_float64_oracle_and_is_immutable() -> None:
    """Tuple-backed inventory matches an independent multi-box NumPy oracle."""
    masses = np.array(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        dtype=np.float64,
    )
    weights = np.array([[2.0, 0.5], [1.5, 2.0]], dtype=np.float64)
    charge = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float64)
    volume = np.array([2.0, 4.0], dtype=np.float64)
    snapshots = tuple(
        value.tobytes() for value in (masses, weights, charge, volume)
    )
    inventory = get_weighted_inventory(masses, weights, charge, volume)
    expected_number = np.sum(weights, axis=1, dtype=np.float64)
    expected_mass = np.einsum("bn,bns->bs", weights, masses, optimize=True)
    expected_charge = np.sum(weights * charge, axis=1, dtype=np.float64)
    npt.assert_allclose(
        inventory.number, expected_number, rtol=1e-12, atol=1e-30
    )
    npt.assert_allclose(inventory.mass, expected_mass, rtol=1e-12, atol=1e-30)
    npt.assert_allclose(
        inventory.charge, expected_charge, rtol=1e-12, atol=1e-30
    )
    npt.assert_allclose(
        inventory.number_per_volume,
        expected_number / volume,
        rtol=1e-12,
        atol=1e-30,
    )
    npt.assert_allclose(
        inventory.mass_per_volume,
        expected_mass / volume[:, None],
        rtol=1e-12,
        atol=1e-30,
    )
    npt.assert_allclose(
        inventory.charge_per_volume,
        expected_charge / volume,
        rtol=1e-12,
        atol=1e-30,
    )
    assert isinstance(inventory, WeightedInventory)
    assert isinstance(inventory.mass, tuple)
    assert isinstance(inventory.mass[0], tuple)
    assert (
        tuple(value.tobytes() for value in (masses, weights, charge, volume))
        == snapshots
    )
    with pytest.raises(FrozenInstanceError):
        inventory.number = ()  # type: ignore[misc]


def test_weighted_inventory_accepts_lists_and_zero_particles() -> None:
    """Array-like values and N=0 retain the nonempty species dimension."""
    inventory = get_weighted_inventory([[[1.0]]], [[2.0]], [[3.0]], [2.0])
    assert inventory.number == (np.float64(2.0),)
    empty = get_weighted_inventory(
        np.empty((1, 0, 2)), np.empty((1, 0)), np.empty((1, 0)), [1.0]
    )
    assert empty.mass == ((np.float64(0.0), np.float64(0.0)),)


@pytest.mark.parametrize(
    ("masses", "weights", "charge", "volume", "message"),
    [
        (np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1)), [1.0], "masses"),
        (
            np.ones((1, 1, 1)),
            np.ones((1, 2)),
            np.ones((1, 1)),
            [1.0],
            "weights",
        ),
        (np.ones((1, 1, 1)), np.ones((1, 1)), np.ones((1, 2)), [1.0], "charge"),
        (
            np.ones((1, 1, 1)),
            np.ones((1, 1)),
            np.ones((1, 1)),
            [[1.0]],
            "volume",
        ),
        (
            np.ones((1, 1, 1)),
            np.ones((1, 1)),
            np.ones((1, 1)),
            [0.0],
            "positive",
        ),
        (
            np.array([[[np.nan]]]),
            np.ones((1, 1)),
            np.ones((1, 1)),
            [1.0],
            "finite",
        ),
        (
            np.ones((1, 1, 1)),
            np.array([[np.nan]]),
            np.ones((1, 1)),
            [1.0],
            "weights must contain only finite values",
        ),
        (
            np.ones((1, 1, 1)),
            np.ones((1, 1)),
            np.array([[np.inf]]),
            [1.0],
            "charge must contain only finite values",
        ),
        (
            np.ones((1, 1, 1)),
            np.ones((1, 1)),
            np.ones((1, 1)),
            [np.inf],
            "finite",
        ),
    ],
)
def test_weighted_inventory_rejects_invalid_shapes_and_values(
    masses: object,
    weights: object,
    charge: object,
    volume: object,
    message: str,
) -> None:
    """Inventory helper rejects malformed and nonphysical converted inputs."""
    with pytest.raises(ValueError, match=message):
        get_weighted_inventory(masses, weights, charge, volume)


def _resampling_particles(boxes: int = 1) -> ParticleData:
    """Create explicit float64 active and inactive fixed-capacity particles."""
    masses = np.array(
        [[[1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 4.0], [0.0, 0.0]]],
        dtype=np.float64,
    )
    concentration = np.array([[1.0, 2.0, 3.0, 4.0, 0.0]], dtype=np.float64)
    charge = np.array([[1.0, -2.0, 3.0, -4.0, 0.0]], dtype=np.float64)
    return ParticleData(
        masses=np.repeat(masses, boxes, axis=0),
        concentration=np.repeat(concentration, boxes, axis=0),
        charge=np.repeat(charge, boxes, axis=0),
        density=np.array([1000.0, 2000.0], dtype=np.float64),
        volume=np.ones(boxes, dtype=np.float64),
    )


def _resampling_p1(boxes: int = 1) -> ExhaustionPlan:
    """Create P1 deferred-resampling records releasing two active slots."""
    free_indices = np.full((boxes, 5), -1, dtype=np.int32)
    free_indices[:, 0] = 0
    return resolve_exhaustion(
        ExhaustionInputs(
            requested_count=np.full(boxes, 3, dtype=np.int32),
            free_count=np.full(boxes, 1, dtype=np.int32),
            resampling_releasable_count=np.full(boxes, 2, dtype=np.int32),
            free_indices=free_indices,
        ),
        ExhaustionControls(),
    )


def test_resampling_plan_is_deterministic_detached_and_conservative() -> None:
    """P2 equal strata conserve represented inventory and detach input state."""
    particles = _resampling_particles()
    plan = plan_resampling(particles, _resampling_p1())
    repeat = plan_resampling(particles, _resampling_p1())
    assert plan == repeat
    box_plan = plan.box_plans[0]
    assert box_plan.retained_indices == (0, 1)
    assert box_plan.released_indices == (2, 3)
    before = get_weighted_inventory(
        particles.masses,
        particles.concentration,
        particles.charge,
        particles.volume,
    )
    particles.masses[:] = 9.0
    assert plan.box_plans[0] == box_plan

    target = _resampling_particles()
    mass_id, concentration_id, charge_id = (
        id(target.masses),
        id(target.concentration),
        id(target.charge),
    )
    assert apply_resampling(target, plan) is target
    assert (id(target.masses), id(target.concentration), id(target.charge)) == (
        mass_id,
        concentration_id,
        charge_id,
    )
    after = get_weighted_inventory(
        target.masses, target.concentration, target.charge, target.volume
    )
    npt.assert_allclose(after.number, before.number, rtol=1e-12, atol=1e-30)
    npt.assert_allclose(after.mass, before.mass, rtol=1e-12, atol=1e-30)
    npt.assert_allclose(after.charge, before.charge, rtol=1e-12, atol=1e-30)
    npt.assert_array_equal(target.masses[0, 2:], 0.0)
    npt.assert_array_equal(target.concentration[0, 2:], 0.0)
    npt.assert_array_equal(target.charge[0, 2:], 0.0)


def test_resampling_multibox_and_zero_release_are_fixed_capacity() -> None:
    """P2 preserves untouched boxes and makes zero-release plans write-free."""
    particles = _resampling_particles(2)
    p1 = _resampling_p1(2)
    zero = ExhaustionBoxPlan(
        0, 0, 0, 0, POLICY_RESAMPLE_DEFERRED, (), (), float("nan")
    )
    plan = plan_resampling(
        particles,
        ExhaustionPlan((p1.box_plans[0], zero)),
    )
    assert plan.box_plans[1] == _ResamplingBoxPlan((), (), (), (), ())
    before_second = tuple(
        field[1].tobytes()
        for field in (
            particles.masses,
            particles.concentration,
            particles.charge,
        )
    )
    apply_resampling(particles, plan)
    assert before_second == tuple(
        field[1].tobytes()
        for field in (
            particles.masses,
            particles.concentration,
            particles.charge,
        )
    )
    assert particles.masses.shape == (2, 5, 2)


def test_resampling_rejects_invalid_state_and_atomic_later_plan_failure() -> (
    None
):
    """Planning and all-box apply preflight reject without partial mutation."""
    particles = _resampling_particles(2)
    particles.masses[0, 4, 0] = 1.0
    snapshot = particles.masses.tobytes()
    with pytest.raises(ValueError, match="inactive"):
        plan_resampling(particles, _resampling_p1(2))
    assert particles.masses.tobytes() == snapshot

    particles = _resampling_particles(2)
    valid = plan_resampling(particles, _resampling_p1(2))
    malformed = _ResamplingPlan(
        (
            valid.box_plans[0],
            _ResamplingBoxPlan(
                (9,),
                (),
                ((np.float64(1.0), np.float64(1.0)),),
                (np.float64(1.0),),
                (np.float64(0.0),),
            ),
        )
    )
    snapshot_fields: tuple[bytes, ...] = tuple(
        field.tobytes()
        for field in (
            particles.masses,
            particles.concentration,
            particles.charge,
        )
    )
    with pytest.raises(ValueError, match="out of range"):
        apply_resampling(particles, malformed)
    assert snapshot_fields == tuple(
        field.tobytes()
        for field in (
            particles.masses,
            particles.concentration,
            particles.charge,
        )
    )


def test_resampling_apply_rejects_stale_active_slots_without_mutation() -> None:
    """Apply rejects plans that would overwrite a pre-existing inactive slot."""
    particles = _resampling_particles()
    plan = plan_resampling(particles, _resampling_p1())
    particles.concentration[0, 1] = 0.0
    particles.masses[0, 1] = 0.0
    particles.charge[0, 1] = 0.0
    snapshot = tuple(
        field.tobytes()
        for field in (
            particles.masses,
            particles.concentration,
            particles.charge,
            particles.density,
            particles.volume,
        )
    )
    with pytest.raises(ValueError, match="exactly cover current active slots"):
        apply_resampling(particles, plan)
    assert snapshot == tuple(
        field.tobytes()
        for field in (
            particles.masses,
            particles.concentration,
            particles.charge,
            particles.density,
            particles.volume,
        )
    )


def test_resampling_equal_strata_matches_independent_weighted_oracle() -> None:
    """P2 assigns crossing source intervals to equal-number strata."""
    particles = _resampling_particles()
    plan = plan_resampling(particles, _resampling_p1())
    box_plan = plan.box_plans[0]

    # Radius/fraction ordering is rows 1, 0, 3, 2. The two equal strata each
    # carry five represented particles and cross source rows.
    expected_mass = np.array([[0.2, 2.4], [1.8, 1.6]], dtype=np.float64)
    expected_charge = np.array([-2.2, 0.2], dtype=np.float64)
    npt.assert_allclose(
        box_plan.replacement_masses,
        expected_mass,
        rtol=1e-12,
        atol=1e-30,
    )
    npt.assert_allclose(
        box_plan.replacement_concentrations,
        [5.0, 5.0],
        rtol=1e-12,
        atol=1e-30,
    )
    npt.assert_allclose(
        box_plan.replacement_charges,
        expected_charge,
        rtol=1e-12,
        atol=1e-30,
    )


@pytest.mark.parametrize(
    "bound",
    [0, np.float64(0.0), float("nan"), float("inf"), -1.0],
)
def test_resampling_rejects_non_exact_or_nonfinite_diagnostic_bounds(
    bound: object,
) -> None:
    """P2 bounds are strict finite Python floats before particle inspection."""
    particles = _resampling_particles()
    snapshot = tuple(
        field.tobytes()
        for field in (
            particles.masses,
            particles.concentration,
            particles.charge,
            particles.density,
            particles.volume,
        )
    )
    with pytest.raises((TypeError, ValueError), match="radius_cubed"):
        plan_resampling(
            particles,
            _resampling_p1(),
            radius_cubed_relative_error=bound,  # type: ignore[arg-type]
        )
    assert snapshot == tuple(
        field.tobytes()
        for field in (
            particles.masses,
            particles.concentration,
            particles.charge,
            particles.density,
            particles.volume,
        )
    )


def test_resampling_rejects_malformed_p1_record_without_mutation() -> None:
    """P2 consumes valid P1 sentinels rather than resolving policy again."""
    particles = _resampling_particles()
    invalid_p1 = ExhaustionPlan(
        (
            ExhaustionBoxPlan(
                3,
                3,
                2,
                2,
                POLICY_RESAMPLE_DEFERRED,
                (),
                (),
                1.0,
            ),
        )
    )
    snapshot = particles.masses.tobytes()
    with pytest.raises(ValueError, match="deferred resampling P1 sentinel"):
        plan_resampling(particles, invalid_p1)
    assert particles.masses.tobytes() == snapshot


def test_resampling_apply_rejects_overlapping_plan_without_mutation() -> None:
    """Apply validates disjoint remap indices before its commit assignments."""
    particles = _resampling_particles()
    invalid_plan = _ResamplingPlan(
        (
            _ResamplingBoxPlan(
                (0,),
                (0,),
                ((np.float64(1.0), np.float64(1.0)),),
                (np.float64(1.0),),
                (np.float64(0.0),),
            ),
        )
    )
    snapshot = tuple(
        field.tobytes()
        for field in (
            particles.masses,
            particles.concentration,
            particles.charge,
        )
    )
    with pytest.raises(ValueError, match="disjoint"):
        apply_resampling(particles, invalid_plan)
    assert snapshot == tuple(
        field.tobytes()
        for field in (
            particles.masses,
            particles.concentration,
            particles.charge,
        )
    )
