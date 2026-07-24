"""Read-only particle-capacity exhaustion planning.

P1 consumes fixed-shape capacity sidecars and creates an immutable, later
commit-boundary plan.  It validates every box before resolving any box, writes
no caller state, and can therefore be retried safely with corrected sidecars.
The exact requested/admitted and required-release counts, policy code, and
activation-prefix identity are diagnostics.  Release tuples are empty and
``float("nan")`` is the no-scale sentinel in P1; ``-1`` is the unused-index
sentinel.

Weighted inventory is calculated in float64 as ``sum(weights)``,
``sum(weights[..., None] * masses, axis=particle)``, and
``sum(weights * charge)`` (without materializing the weighted-mass product).
Intensive values divide these quantities by volume.  A later unscaled commit
must equal ``pre_state + source`` and a scaling commit must equal
``scale * pre_state + source``.  No radius-cubed or other moment is exact in
P1; any later preservation claim must specify its moment, domain, and
tolerance before a commit can rely on it.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

POLICY_ACTIVATE = 0
POLICY_RESAMPLE_DEFERRED = 1
POLICY_SCALE_DEFERRED = 2


@dataclass(frozen=True)
class ExhaustionControls:
    """Control which deferred policies may represent exhausted capacity.

    Attributes:
        resampling: Whether a sufficiently releasable box may use the deferred
            resampling policy.
        representative_volume_scaling: Whether a box not represented by
            resampling may use the deferred scaling policy.

    Raises:
        TypeError: If either control is not an exact Python ``bool``.
    """

    resampling: bool = True
    representative_volume_scaling: bool = False

    def __post_init__(self) -> None:
        """Reject bool-like values so policy controls remain unambiguous."""
        for name, value in (
            ("resampling", self.resampling),
            (
                "representative_volume_scaling",
                self.representative_volume_scaling,
            ),
        ):
            if type(value) is not bool:
                raise TypeError(f"{name} must be an exact Python bool")


@dataclass(frozen=True)
class ExhaustionInputs:
    """Hold fixed-shape capacity sidecars for exhaustion planning.

    Attributes:
        requested_count: Requested activation count per box, as int32 values
            with shape ``(B,)``.
        free_count: Available free-slot count per box, as int32 values with
            shape ``(B,)``.
        resampling_releasable_count: Count per box that deferred resampling
            could release, as int32 values with shape ``(B,)``.
        free_indices: Ascending free-slot prefixes followed by ``-1`` unused
            sentinels, as int32 values with shape ``(B, N)``.

    Validation occurs when :func:`resolve_exhaustion` consumes this record;
    record construction intentionally performs no coercion or mutation.
    """

    requested_count: NDArray[np.int32]
    free_count: NDArray[np.int32]
    resampling_releasable_count: NDArray[np.int32]
    free_indices: NDArray[np.int32]


@dataclass(frozen=True)
class ExhaustionBoxPlan:
    """Describe one immutable capacity decision at the later commit boundary.

    Attributes:
        requested_count: Requested activation count for the box.
        admitted_count: Count admitted by this plan; equals ``requested_count``
            in P1.
        required_release_count: Additional slots needed after free capacity is
            exhausted.
        releasable_count: Capacity reported as releasable by resampling.
        policy_code: ``POLICY_ACTIVATE``, ``POLICY_RESAMPLE_DEFERRED``, or
            ``POLICY_SCALE_DEFERRED``.
        activation_indices: Exact free-index prefix for activation plans; empty
            for deferred plans.
        release_indices: Empty in P1 because release selection is deferred.
        scale_factor: ``float("nan")`` in P1 because scaling is deferred.
    """

    requested_count: int
    admitted_count: int
    required_release_count: int
    releasable_count: int
    policy_code: int
    activation_indices: tuple[int, ...]
    release_indices: tuple[int, ...]
    scale_factor: float


@dataclass(frozen=True)
class ExhaustionPlan:
    """Hold immutable per-box decisions for one later commit boundary.

    Attributes:
        box_plans: One plan per validated input box, in input-box order.
    """

    box_plans: tuple[ExhaustionBoxPlan, ...]


@dataclass(frozen=True)
class WeightedInventory:
    """Hold tuple-backed float64 extensive and intensive inventories.

    Attributes:
        number: Weighted particle number, ``sum(weights)``, per box.
        mass: Weighted mass, ``sum(weights * masses)``, per box and species.
        charge: Weighted charge, ``sum(weights * charge)``, per box.
        number_per_volume: Weighted number divided by box volume.
        mass_per_volume: Weighted species mass divided by box volume.
        charge_per_volume: Weighted charge divided by box volume.
    """

    number: tuple[np.float64, ...]
    mass: tuple[tuple[np.float64, ...], ...]
    charge: tuple[np.float64, ...]
    number_per_volume: tuple[np.float64, ...]
    mass_per_volume: tuple[tuple[np.float64, ...], ...]
    charge_per_volume: tuple[np.float64, ...]


def _validate_sidecar_schema(inputs: ExhaustionInputs) -> None:
    """Validate sidecar array types, dtypes, and ranks without coercion."""
    arrays = (
        ("requested_count", inputs.requested_count, 1),
        ("free_count", inputs.free_count, 1),
        (
            "resampling_releasable_count",
            inputs.resampling_releasable_count,
            1,
        ),
        ("free_indices", inputs.free_indices, 2),
    )
    for name, values, rank in arrays:
        if not isinstance(values, np.ndarray):
            raise TypeError(f"{name} must be a numpy array")
        if values.dtype != np.dtype(np.int32):
            raise TypeError(f"{name} must have dtype int32")
        if values.ndim != rank:
            raise ValueError(f"{name} must have rank {rank}")


def _validate_sidecar_shapes(inputs: ExhaustionInputs) -> tuple[int, int]:
    """Validate cross-box sidecar shapes and return box and particle counts."""
    box_count = inputs.requested_count.shape[0]
    if (
        inputs.free_count.shape != (box_count,)
        or inputs.resampling_releasable_count.shape != (box_count,)
        or inputs.free_indices.shape[0] != box_count
    ):
        raise ValueError("sidecars must have matching box dimensions")

    particle_count = inputs.free_indices.shape[1]
    return box_count, particle_count


def _validate_capacity_counts(
    inputs: ExhaustionInputs,
    particle_count: int,
) -> None:
    """Validate count ranges against fixed particle capacity."""
    arrays = (
        ("requested_count", inputs.requested_count),
        ("free_count", inputs.free_count),
        ("resampling_releasable_count", inputs.resampling_releasable_count),
    )
    for name, values in arrays:
        if np.any(values < 0):
            raise ValueError(f"{name} must be nonnegative")
        if np.any(values > particle_count):
            raise ValueError(f"{name} exceeds particle capacity")


def _validate_free_indices(
    inputs: ExhaustionInputs,
    particle_count: int,
) -> None:
    """Validate free-index prefixes and unused suffix sentinels per box."""
    for box_index, free_count in enumerate(inputs.free_count):
        count = int(free_count)
        prefix = inputs.free_indices[box_index, :count]
        suffix = inputs.free_indices[box_index, count:]
        if np.any(prefix < 0) or np.any(prefix >= particle_count):
            raise ValueError(
                "free_indices prefix contains an out-of-range index"
            )
        if count > 1 and np.any(prefix[1:] <= prefix[:-1]):
            raise ValueError("free_indices prefix must be strictly ascending")
        if np.any(suffix != -1):
            raise ValueError("free_indices unused suffix must contain -1")


def _validate_sidecars(inputs: ExhaustionInputs) -> tuple[int, int]:
    """Validate all sidecars without coercion, sorting, copying, or mutation."""
    _validate_sidecar_schema(inputs)
    box_count, particle_count = _validate_sidecar_shapes(inputs)
    _validate_capacity_counts(inputs, particle_count)
    _validate_free_indices(inputs, particle_count)
    return box_count, particle_count


def resolve_exhaustion(
    inputs: ExhaustionInputs,
    controls: ExhaustionControls,
) -> ExhaustionPlan:
    """Resolve immutable capacity decisions after complete sidecar validation.

    The resolver validates every input box before creating a returned plan.
    Activation uses the exact requested free-index prefix. Exhausted capacity
    uses sufficiently releasable resampling before enabled scaling; both remain
    deferred and therefore have empty release indices and a ``nan`` scale.

    Args:
        inputs: Fixed-shape, int32 capacity sidecars.
        controls: Strict policy enablement controls.

    Returns:
        A tuple-backed plan with activation-prefix identity or deferred policy.

    Raises:
        TypeError: If either public record has the wrong type.
        ValueError: If sidecars are invalid or no enabled policy can represent
            exhausted capacity.
    """
    if not isinstance(controls, ExhaustionControls):
        raise TypeError("controls must be an ExhaustionControls")
    if not isinstance(inputs, ExhaustionInputs):
        raise TypeError("inputs must be an ExhaustionInputs")

    box_count, _ = _validate_sidecars(inputs)
    box_plans: list[ExhaustionBoxPlan] = []
    for box_index in range(box_count):
        requested = int(inputs.requested_count[box_index])
        free = int(inputs.free_count[box_index])
        releasable = int(inputs.resampling_releasable_count[box_index])
        if requested <= free:
            activation_indices = tuple(
                int(index)
                for index in inputs.free_indices[box_index, :requested]
            )
            box_plans.append(
                ExhaustionBoxPlan(
                    requested_count=requested,
                    admitted_count=requested,
                    required_release_count=0,
                    releasable_count=releasable,
                    policy_code=POLICY_ACTIVATE,
                    activation_indices=activation_indices,
                    release_indices=(),
                    scale_factor=float("nan"),
                )
            )
            continue

        required_release = requested - free
        if controls.resampling and releasable >= required_release:
            policy_code = POLICY_RESAMPLE_DEFERRED
        elif controls.representative_volume_scaling:
            policy_code = POLICY_SCALE_DEFERRED
        else:
            raise ValueError(
                "exhaustion policy cannot represent requested capacity"
            )
        box_plans.append(
            ExhaustionBoxPlan(
                requested_count=requested,
                admitted_count=requested,
                required_release_count=required_release,
                releasable_count=releasable,
                policy_code=policy_code,
                activation_indices=(),
                release_indices=(),
                scale_factor=float("nan"),
            )
        )
    return ExhaustionPlan(box_plans=tuple(box_plans))


def get_weighted_inventory(
    masses: object,
    weights: object,
    charge: object,
    volume: object,
) -> WeightedInventory:
    """Calculate float64 weighted inventories without retaining caller arrays.

    Array-like inputs are converted to float64. The mass reduction avoids a
    full weighted-mass broadcast product. Returned tuples do not retain caller
    array storage.

    Args:
        masses: Array-like particle masses with shape ``(B, N, S)``.
        weights: Array-like particle weights with shape ``(B, N)``.
        charge: Array-like particle charge with shape ``(B, N)``.
        volume: Array-like box volumes with shape ``(B,)``.

    Returns:
        Tuple-backed extensive and intensive float64 inventories.

    Raises:
        ValueError: If converted inputs have invalid shape or physical values.
    """
    mass_values = np.asarray(masses, dtype=np.float64)
    weight_values = np.asarray(weights, dtype=np.float64)
    charge_values = np.asarray(charge, dtype=np.float64)
    volume_values = np.asarray(volume, dtype=np.float64)
    if mass_values.ndim != 3:
        raise ValueError("masses must have shape (B, N, S)")
    box_count, particle_count, _ = mass_values.shape
    if weight_values.shape != (box_count, particle_count):
        raise ValueError("weights must have shape (B, N)")
    if charge_values.shape != (box_count, particle_count):
        raise ValueError("charge must have shape (B, N)")
    if volume_values.shape != (box_count,):
        raise ValueError("volume must have shape (B,)")
    for name, values in (
        ("masses", mass_values),
        ("weights", weight_values),
        ("charge", charge_values),
        ("volume", volume_values),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")
    if np.any(volume_values <= 0):
        raise ValueError("volume must be positive")

    number = np.sum(weight_values, axis=1, dtype=np.float64)
    mass = np.einsum(
        "bn,bns->bs",
        weight_values,
        mass_values,
        dtype=np.float64,
        optimize=True,
    )
    total_charge = np.sum(
        weight_values * charge_values, axis=1, dtype=np.float64
    )
    return WeightedInventory(
        number=tuple(np.float64(value) for value in number),
        mass=tuple(tuple(np.float64(value) for value in row) for row in mass),
        charge=tuple(np.float64(value) for value in total_charge),
        number_per_volume=tuple(
            np.float64(value) for value in number / volume_values
        ),
        mass_per_volume=tuple(
            tuple(np.float64(value) for value in row)
            for row in mass / volume_values[:, None]
        ),
        charge_per_volume=tuple(
            np.float64(value) for value in total_charge / volume_values
        ),
    )
