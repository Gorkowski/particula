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
    _riemer_diversity,
    apply_representative_volume_scaling,
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


def test_resampling_apply_rejects_stale_source_and_forged_replacement() -> None:
    """P2 binds replacements to source values and validates conservation."""
    particles = _resampling_particles()
    plan = plan_resampling(particles, _resampling_p1())
    particles.masses[0, 0, 0] = 1.5
    snapshot = tuple(
        field.tobytes()
        for field in (
            particles.masses,
            particles.concentration,
            particles.charge,
        )
    )
    with pytest.raises(ValueError, match="source state is stale"):
        apply_resampling(particles, plan)
    assert snapshot == tuple(
        field.tobytes()
        for field in (
            particles.masses,
            particles.concentration,
            particles.charge,
        )
    )

    particles = _resampling_particles()
    valid = plan_resampling(particles, _resampling_p1()).box_plans[0]
    forged = _ResamplingPlan(
        (
            _ResamplingBoxPlan(
                valid.retained_indices,
                valid.released_indices,
                ((np.float64(9.0), np.float64(9.0)),) * 2,
                valid.replacement_concentrations,
                valid.replacement_charges,
                valid.source_concentrations,
                valid.source_masses,
                valid.source_charges,
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
    with pytest.raises(ValueError, match="does not conserve inventory"):
        apply_resampling(particles, forged)
    assert snapshot == tuple(
        field.tobytes()
        for field in (
            particles.masses,
            particles.concentration,
            particles.charge,
        )
    )


@pytest.mark.parametrize("field_name", ["concentration", "charge"])
def test_resampling_apply_rejects_stale_source_sidecars_without_mutation(
    field_name: str,
) -> None:
    """Apply binds remaps to the planned concentration and charge sidecars."""
    particles = _resampling_particles()
    plan = plan_resampling(particles, _resampling_p1())
    field = getattr(particles, field_name)
    field[0, 0] += 0.5
    snapshot = tuple(
        values.tobytes()
        for values in (
            particles.masses,
            particles.concentration,
            particles.charge,
        )
    )

    with pytest.raises(ValueError, match="source state is stale"):
        apply_resampling(particles, plan)

    assert snapshot == tuple(
        values.tobytes()
        for values in (
            particles.masses,
            particles.concentration,
            particles.charge,
        )
    )


def test_riemer_diversity_uses_particle_mixing_not_bulk_inventory() -> None:
    """Equal bulk inventory can have distinct particle mixing diversity."""
    concentrations = np.array([1.0, 1.0], dtype=np.float64)
    externally_mixed = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    internally_mixed = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    npt.assert_allclose(
        _riemer_diversity(externally_mixed, concentrations), 1.0
    )
    assert _riemer_diversity(internally_mixed, concentrations) > 1.9
    assert (
        _riemer_diversity(np.zeros((2, 2), dtype=np.float64), concentrations)
        == 0.0
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


def test_resampling_rejects_p1_release_count_above_requested() -> None:
    """P2 rejects forged P1 release counts before remap planning."""
    particles = _resampling_particles()
    invalid_p1 = ExhaustionPlan(
        (
            ExhaustionBoxPlan(
                1,
                1,
                2,
                2,
                POLICY_RESAMPLE_DEFERRED,
                (),
                (),
                float("nan"),
            ),
        )
    )
    with pytest.raises(ValueError, match="exceeds requested"):
        plan_resampling(particles, invalid_p1)


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


def test_representative_volume_scaling_scales_only_selected_rows() -> None:
    """P4 scales selected extensive state and preserves protected siblings."""
    particles = _resampling_particles(2)
    particles.volume[:] = [4.0, 5.0]
    demand = np.array([3.0, 2.0], dtype=np.float64)
    flags = np.array([True, False], dtype=np.bool_)
    requested = np.array([0.5, 0.8], dtype=np.float64)
    minimum = np.array([0.25, 0.5], dtype=np.float64)
    minimum_volume = np.array([1.0, 1.0], dtype=np.float64)
    resolved = np.zeros(2, dtype=np.float64)
    before = tuple(
        field.copy()
        for field in (particles.masses, particles.charge, particles.density)
    )
    original_second = particles.concentration[1].copy()

    result = apply_representative_volume_scaling(
        particles, demand, flags, requested, minimum, minimum_volume, resolved
    )

    assert result == (particles, demand, resolved)
    npt.assert_allclose(particles.volume, [2.0, 5.0])
    npt.assert_allclose(demand, [1.5, 2.0])
    npt.assert_allclose(resolved, [0.5, 1.0])
    npt.assert_allclose(particles.concentration[0], [0.5, 1.0, 1.5, 2.0, 0.0])
    npt.assert_array_equal(particles.concentration[1], original_second)
    for actual, expected in zip(
        (particles.masses, particles.charge, particles.density),
        before,
        strict=True,
    ):
        npt.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("demand", "flags"),
    [([2.0, 3.0], [False, False]), ([0.0, 0.0], [True, True])],
)
def test_representative_volume_scaling_no_selected_rows_writes_diagnostic_only(
    demand: list[float], flags: list[bool]
) -> None:
    """P4 emits one diagnostics for all-false and zero-demand selections."""
    particles = _resampling_particles(2)
    demand_values = np.array(demand, dtype=np.float64)
    flags_values = np.array(flags, dtype=np.bool_)
    requested = np.array([0.5, 0.5], dtype=np.float64)
    minimum = np.array([0.25, 0.25], dtype=np.float64)
    minimum_volume = np.array([0.1, 0.1], dtype=np.float64)
    resolved = np.zeros(2, dtype=np.float64)
    snapshot = tuple(
        field.tobytes()
        for field in (
            particles.masses,
            particles.concentration,
            particles.charge,
            particles.density,
            particles.volume,
            demand_values,
        )
    )
    apply_representative_volume_scaling(
        particles,
        demand_values,
        flags_values,
        requested,
        minimum,
        minimum_volume,
        resolved,
    )
    assert np.array_equal(resolved, np.ones(2))
    assert snapshot == tuple(
        field.tobytes()
        for field in (
            particles.masses,
            particles.concentration,
            particles.charge,
            particles.density,
            particles.volume,
            demand_values,
        )
    )


def test_representative_volume_scaling_selected_unity_scale_is_exact_noop() -> (
    None
):
    """A selected row with scale one preserves data and reports one exactly."""
    particles = _resampling_particles()
    demand = np.array([2.0], dtype=np.float64)
    flags = np.array([True], dtype=np.bool_)
    requested = np.array([1.0], dtype=np.float64)
    minimum = np.array([1.0], dtype=np.float64)
    minimum_volume = np.array([0.1], dtype=np.float64)
    resolved = np.array([9.0], dtype=np.float64)
    snapshot = tuple(
        values.tobytes()
        for values in (
            particles.masses,
            particles.concentration,
            particles.charge,
            particles.density,
            particles.volume,
            demand,
        )
    )

    returned = apply_representative_volume_scaling(
        particles, demand, flags, requested, minimum, minimum_volume, resolved
    )

    assert returned[0] is particles
    assert returned[1] is demand
    assert returned[2] is resolved
    assert snapshot == tuple(
        values.tobytes()
        for values in (
            particles.masses,
            particles.concentration,
            particles.charge,
            particles.density,
            particles.volume,
            demand,
        )
    )
    npt.assert_array_equal(resolved, [1.0])


def test_representative_volume_scaling_rejects_invalid_volume_atomically() -> (
    None
):
    """A later infeasible P4 row prevents earlier selected-row mutation."""
    particles = _resampling_particles(2)
    demand = np.array([2.0, 2.0], dtype=np.float64)
    flags = np.array([True, True], dtype=np.bool_)
    requested = np.array([0.5, 0.5], dtype=np.float64)
    minimum = np.array([0.25, 0.25], dtype=np.float64)
    minimum_volume = np.array([0.1, 1.0], dtype=np.float64)
    resolved = np.array([9.0, 9.0], dtype=np.float64)
    snapshot = tuple(
        field.tobytes()
        for field in (
            particles.masses,
            particles.concentration,
            particles.charge,
            particles.density,
            particles.volume,
            demand,
            flags,
            requested,
            minimum,
            minimum_volume,
            resolved,
        )
    )
    with pytest.raises(ValueError, match="scaled volume"):
        apply_representative_volume_scaling(
            particles,
            demand,
            flags,
            requested,
            minimum,
            minimum_volume,
            resolved,
        )
    assert snapshot == tuple(
        field.tobytes()
        for field in (
            particles.masses,
            particles.concentration,
            particles.charge,
            particles.density,
            particles.volume,
            demand,
            flags,
            requested,
            minimum,
            minimum_volume,
            resolved,
        )
    )


def test_representative_volume_scaling_preserves_intensities_and_surface() -> (
    None
):
    """P4 scales selected extensive inventories while preserving intensities."""
    import particula.particles as particles_package

    particles = _resampling_particles()
    particles.volume[:] = 4.0
    demand = np.array([2.0], dtype=np.float64)
    flags = np.array([True], dtype=np.bool_)
    requested = np.array([0.5], dtype=np.float64)
    minimum = np.array([0.25], dtype=np.float64)
    minimum_volume = np.array([1.0], dtype=np.float64)
    resolved = np.zeros(1, dtype=np.float64)
    before = get_weighted_inventory(
        particles.masses,
        particles.concentration,
        particles.charge,
        particles.volume,
    )

    returned = apply_representative_volume_scaling(
        particles, demand, flags, requested, minimum, minimum_volume, resolved
    )
    after = get_weighted_inventory(
        particles.masses,
        particles.concentration,
        particles.charge,
        particles.volume,
    )

    assert returned[0] is particles
    assert returned[1] is demand
    assert returned[2] is resolved
    assert not hasattr(particles_package, "apply_representative_volume_scaling")
    npt.assert_allclose(after.number, np.asarray(before.number) * 0.5)
    npt.assert_allclose(after.mass, np.asarray(before.mass) * 0.5)
    npt.assert_allclose(after.charge, np.asarray(before.charge) * 0.5)
    npt.assert_allclose(after.number_per_volume, before.number_per_volume)
    npt.assert_allclose(after.mass_per_volume, before.mass_per_volume)
    npt.assert_allclose(after.charge_per_volume, before.charge_per_volume)
    npt.assert_allclose(demand / particles.volume, [0.5])


def test_representative_volume_scaling_accepts_empty_boxes() -> None:
    """P4 accepts B=0 while preserving every supplied empty array identity."""
    particles = ParticleData(
        masses=np.empty((0, 1, 1), dtype=np.float64),
        concentration=np.empty((0, 1), dtype=np.float64),
        charge=np.empty((0, 1), dtype=np.float64),
        density=np.array([1.0], dtype=np.float64),
        volume=np.empty(0, dtype=np.float64),
    )
    demand = np.empty(0, dtype=np.float64)
    flags = np.empty(0, dtype=np.bool_)
    requested = np.empty(0, dtype=np.float64)
    minimum = np.empty(0, dtype=np.float64)
    minimum_volume = np.empty(0, dtype=np.float64)
    resolved = np.empty(0, dtype=np.float64)

    returned = apply_representative_volume_scaling(
        particles, demand, flags, requested, minimum, minimum_volume, resolved
    )

    assert returned == (particles, demand, resolved)


def test_representative_volume_scaling_rejects_factor_bounds_atomically() -> (
    None
):
    """P4 validates unselected factor bounds before writing its diagnostic."""
    particles = _resampling_particles()
    demand = np.array([0.0], dtype=np.float64)
    flags = np.array([False], dtype=np.bool_)
    requested = np.array([0.4], dtype=np.float64)
    minimum = np.array([0.5], dtype=np.float64)
    minimum_volume = np.array([0.1], dtype=np.float64)
    resolved = np.array([9.0], dtype=np.float64)
    snapshot = tuple(
        value.tobytes()
        for value in (
            particles.masses,
            particles.concentration,
            particles.charge,
            particles.density,
            particles.volume,
            demand,
            flags,
            requested,
            minimum,
            minimum_volume,
            resolved,
        )
    )

    with pytest.raises(ValueError, match="0 < minimum <= requested <= 1"):
        apply_representative_volume_scaling(
            particles,
            demand,
            flags,
            requested,
            minimum,
            minimum_volume,
            resolved,
        )

    assert snapshot == tuple(
        value.tobytes()
        for value in (
            particles.masses,
            particles.concentration,
            particles.charge,
            particles.density,
            particles.volume,
            demand,
            flags,
            requested,
            minimum,
            minimum_volume,
            resolved,
        )
    )


@pytest.mark.parametrize(
    ("name", "replacement", "error", "message"),
    [
        ("provisional_source_demand", [1.0], TypeError, "numpy array"),
        ("scaling_required", np.array([1], dtype=np.int32), TypeError, "dtype"),
        ("requested_scale", np.ones((1, 1)), ValueError, "rank 1"),
        ("minimum_scale", np.ones(2), ValueError, "shape"),
    ],
)
def test_representative_volume_scaling_rejects_sidecar_schema_atomically(
    name: str,
    replacement: object,
    error: type[Exception],
    message: str,
) -> None:
    """P4 reports stable sidecar schema failures before any caller write."""
    particles = _resampling_particles()
    sidecars: dict[str, object] = {
        "provisional_source_demand": np.array([1.0], dtype=np.float64),
        "scaling_required": np.array([True], dtype=np.bool_),
        "requested_scale": np.array([0.5], dtype=np.float64),
        "minimum_scale": np.array([0.25], dtype=np.float64),
        "minimum_volume": np.array([0.1], dtype=np.float64),
        "resolved_scale": np.array([9.0], dtype=np.float64),
    }
    sidecars[name] = replacement
    snapshot = tuple(
        value.tobytes()
        for value in (
            particles.masses,
            particles.concentration,
            particles.charge,
            particles.density,
            particles.volume,
            *(
                value
                for value in sidecars.values()
                if isinstance(value, np.ndarray)
            ),
        )
    )

    with pytest.raises(error, match=message):
        apply_representative_volume_scaling(particles, **sidecars)  # type: ignore[arg-type]

    assert snapshot == tuple(
        value.tobytes()
        for value in (
            particles.masses,
            particles.concentration,
            particles.charge,
            particles.density,
            particles.volume,
            *(
                value
                for value in sidecars.values()
                if isinstance(value, np.ndarray)
            ),
        )
    )


@pytest.mark.parametrize(
    ("masses", "message"),
    [
        (np.empty((1, 0, 1), dtype=np.float64), "particle capacity"),
        (np.empty((1, 1, 0), dtype=np.float64), "species capacity"),
    ],
)
def test_representative_volume_scaling_rejects_zero_capacity(
    masses: np.ndarray,
    message: str,
) -> None:
    """P4 distinguishes valid empty boxes from rejected zero capacities."""
    boxes, particles_count, species = masses.shape
    particles = ParticleData(
        masses=masses,
        concentration=np.zeros((boxes, particles_count), dtype=np.float64),
        charge=np.zeros((boxes, particles_count), dtype=np.float64),
        density=np.ones(species, dtype=np.float64),
        volume=np.ones(boxes, dtype=np.float64),
    )
    with pytest.raises(ValueError, match=message):
        apply_representative_volume_scaling(
            particles,
            np.ones(boxes, dtype=np.float64),
            np.ones(boxes, dtype=np.bool_),
            np.ones(boxes, dtype=np.float64),
            np.ones(boxes, dtype=np.float64),
            np.ones(boxes, dtype=np.float64),
            np.zeros(boxes, dtype=np.float64),
        )
