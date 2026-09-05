"""Concrete host-side support for resident benchmark evidence.

This test-support module validates and persists benchmark artifacts without
importing or probing Warp or CUDA, allocating device resources, or changing
resident execution.
"""

# ruff: noqa: C901, E501

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

RESIDENT_BENCHMARK_SCHEMA_VERSION = 1
"""Version of the resident benchmark JSON envelope."""

MAX_TIMING_SAMPLES = 10_000
"""Maximum P1 timing samples accepted for one executed benchmark result."""

PROCESS_ORDER = (
    "communication",
    "environment",
    "gas",
    "condensation",
    "coagulation",
    "dilution",
    "wall_loss",
    "nucleation",
    "diagnostics",
)
"""Canonical ordering for process combinations in resident evidence."""

SUPPORTED_TIMING_MODES = frozenset({"wall_clock", "device_synchronized"})
SUPPORTED_COMMUNICATION = frozenset({"none", "gas", "particles"})
REQUIRED_METADATA_FIELDS = frozenset(
    {
        "timestamp_utc",
        "command",
        "python_version",
        "platform",
        "warp_version",
        "device",
        "synchronization_method",
        "warmup",
        "timestep_count",
        "seed",
        "prepared_signature_digest",
    }
)


class ResidentBenchmarkStatus(str, Enum):
    """Status of one requested resident benchmark measurement."""

    EXECUTED = "executed"
    UNAVAILABLE = "unavailable"
    SKIPPED_BUDGET = "skipped_budget"


def _require_int(value: object, name: str, *, positive: bool = False) -> int:
    """Validate and return a non-bool integer field.

    Args:
        value: Candidate integer value.
        name: Field name used in error messages.
        positive: Require a value greater than zero when true; otherwise
            require a nonnegative value.

    Returns:
        The validated integer.

    Raises:
        TypeError: If ``value`` is not an integer or is a boolean.
        ValueError: If ``value`` violates the requested lower bound.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-bool integer.")
    if (positive and value <= 0) or (not positive and value < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}.")
    return value


def _require_float(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
) -> float:
    """Validate and return a finite numeric field.

    Args:
        value: Candidate numeric value.
        name: Field name used in error messages.
        minimum: Optional inclusive lower bound.

    Returns:
        The value converted to ``float``.

    Raises:
        TypeError: If ``value`` is not a non-boolean number.
        ValueError: If the value is nonfinite or below ``minimum``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number.")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and converted < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return converted


def _validate_shape(value: object, name: str) -> tuple[int, int, int]:
    """Validate a positive three-dimensional benchmark shape.

    Args:
        value: Candidate shape tuple in boxes, particles, and species order.
        name: Field name used in error messages.

    Returns:
        The validated shape as a three-item integer tuple.

    Raises:
        TypeError: If ``value`` is not a three-item tuple of integers.
        ValueError: If any dimension is not positive.
    """
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError(f"{name} must be a three-item tuple.")
    return tuple(
        _require_int(item, f"{name}[{index}]", positive=True)
        for index, item in enumerate(value)
    )  # type: ignore[return-value]


def _freeze_mapping(
    value: object, name: str, *, nonempty: bool = False
) -> Mapping[str, Any]:
    """Normalize and recursively freeze a string-keyed mapping.

    Args:
        value: Candidate mapping to validate.
        name: Field name used in error messages.
        nonempty: Require at least one mapping entry when true.

    Returns:
        An immutable, normalized mapping suitable for a frozen record.

    Raises:
        TypeError: If ``value`` is not a mapping or contains unsupported data.
        ValueError: If the mapping is required to be nonempty but is empty.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    if nonempty and not value:
        raise ValueError(f"{name} must not be empty.")
    normalized = _normalize_json(value)
    if not isinstance(
        normalized, dict
    ):  # Defensive: mappings normalize to dicts.
        raise TypeError(f"{name} must be a mapping.")
    return _freeze_json_value(normalized)


def _freeze_json_value(value: Any) -> Any:
    """Recursively make a normalized JSON-compatible value immutable."""
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _validate_metadata(value: object) -> Mapping[str, Any]:
    """Validate the complete stable metadata schema for an artifact."""
    metadata = _freeze_mapping(value, "metadata", nonempty=True)
    if set(metadata) != REQUIRED_METADATA_FIELDS:
        raise ValueError("metadata has invalid fields.")
    for name in (
        "timestamp_utc",
        "command",
        "python_version",
        "platform",
        "synchronization_method",
        "prepared_signature_digest",
    ):
        if not isinstance(metadata[name], str) or not metadata[name]:
            raise ValueError(f"metadata {name} must be a nonempty string.")
    try:
        timestamp = datetime.fromisoformat(
            metadata["timestamp_utc"].replace("Z", "+00:00")
        )
    except ValueError as error:
        raise ValueError("metadata timestamp_utc must be valid UTC.") from error
    if (
        timestamp.tzinfo is None
        or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp)
        or not metadata["timestamp_utc"].endswith("Z")
    ):
        raise ValueError("metadata timestamp_utc must be valid UTC.")
    _require_int(metadata["warmup"], "metadata warmup")
    _require_int(
        metadata["timestep_count"], "metadata timestep_count", positive=True
    )
    _require_int(metadata["seed"], "metadata seed")
    for name in ("warp_version", "device"):
        if not isinstance(metadata[name], Mapping) or not metadata[name]:
            raise ValueError(f"metadata {name} must be a nonempty mapping.")
    return metadata


def _canonical_processes(value: object) -> tuple[str, ...]:
    """Validate the supported process names and their canonical ordering.

    Args:
        value: Candidate nonempty process-name tuple.

    Returns:
        The validated process tuple.

    Raises:
        TypeError: If the value is not a tuple of strings.
        ValueError: If names are unsupported, duplicated, or out of order.
    """
    if not isinstance(value, tuple) or not value:
        raise TypeError("processes must be a nonempty tuple.")
    if any(not isinstance(process, str) for process in value):
        raise TypeError("processes must contain strings.")
    if len(set(value)) != len(value) or any(
        process not in PROCESS_ORDER for process in value
    ):
        raise ValueError("processes must be unique supported process names.")
    expected = tuple(process for process in PROCESS_ORDER if process in value)
    if value != expected:
        raise ValueError("processes must use canonical process ordering.")
    return value


def build_resident_benchmark_case_id(
    *,
    requested_shape: tuple[int, int, int],
    actual_shape: tuple[int, int, int],
    active_fraction: float,
    processes: tuple[str, ...],
    communication: str,
    diagnostics: tuple[str, ...],
    warmup: int,
    timestep_count: int,
    seed: int,
) -> str:
    """Build the canonical, configuration-derived resident case identifier."""
    requested_shape = _validate_shape(requested_shape, "requested_shape")
    actual_shape = _validate_shape(actual_shape, "actual_shape")
    if any(
        actual_item > requested_item
        for actual_item, requested_item in zip(
            actual_shape, requested_shape, strict=True
        )
    ):
        raise ValueError("actual_shape must not exceed requested_shape.")
    active_fraction = _require_float(active_fraction, "active_fraction")
    if not 0.0 <= active_fraction <= 1.0:
        raise ValueError("active_fraction must be in [0, 1].")
    processes = _canonical_processes(processes)
    if not isinstance(communication, str):
        raise TypeError("communication must be a string.")
    if communication not in SUPPORTED_COMMUNICATION:
        raise ValueError("communication must be a supported selection.")
    if not isinstance(diagnostics, tuple):
        raise TypeError("diagnostics must be a tuple.")
    if any(not isinstance(item, str) for item in diagnostics):
        raise TypeError("diagnostics must contain strings.")
    if len(set(diagnostics)) != len(diagnostics):
        raise ValueError("diagnostics must not contain duplicates.")
    _require_int(warmup, "warmup")
    _require_int(timestep_count, "timestep_count", positive=True)
    _require_int(seed, "seed")
    return (
        f"r{requested_shape[0]}x{requested_shape[1]}x{requested_shape[2]}"
        f"-a{actual_shape[0]}x{actual_shape[1]}x{actual_shape[2]}"
        f"-f{active_fraction:.17g}-p{'-'.join(processes)}"
        f"-c{communication}-d{'-'.join(diagnostics) or 'none'}"
        f"-w{warmup}-t{timestep_count}-s{seed}"
    )


@dataclass(frozen=True, slots=True)
class ResidentBenchmarkCase:
    """Validated requested and realizable resident benchmark configuration."""

    case_id: str
    requested_shape: tuple[int, int, int]
    actual_shape: tuple[int, int, int]
    active_fraction: float
    processes: tuple[str, ...]
    communication: str
    diagnostics: tuple[str, ...]
    warmup: int
    timestep_count: int
    seed: int

    def __post_init__(self) -> None:
        """Validate canonical case configuration before it can be used."""
        requested = _validate_shape(self.requested_shape, "requested_shape")
        actual = _validate_shape(self.actual_shape, "actual_shape")
        if any(
            actual_item > requested_item
            for actual_item, requested_item in zip(
                actual, requested, strict=True
            )
        ):
            raise ValueError("actual_shape must not exceed requested_shape.")
        fraction = _require_float(self.active_fraction, "active_fraction")
        if fraction > 1.0:
            raise ValueError("active_fraction must be in [0, 1].")
        processes = _canonical_processes(self.processes)
        canonical_id = build_resident_benchmark_case_id(
            requested_shape=requested,
            actual_shape=actual,
            active_fraction=fraction,
            processes=processes,
            communication=self.communication,
            diagnostics=self.diagnostics,
            warmup=self.warmup,
            timestep_count=self.timestep_count,
            seed=self.seed,
        )
        if not isinstance(self.case_id, str):
            raise TypeError("case_id must be a string.")
        if self.case_id != canonical_id:
            raise ValueError(
                "case_id must exactly match its canonical configuration."
            )


@dataclass(frozen=True, slots=True)
class ResidentTimingSummary:
    """Deterministic summary of nonnegative host timing samples."""

    count: int
    minimum: float
    median: float
    mean: float
    p95: float

    def __post_init__(self) -> None:
        """Validate independently constructed timing summary fields."""
        _require_int(self.count, "count", positive=True)
        for name in ("minimum", "median", "mean", "p95"):
            _require_float(getattr(self, name), name, minimum=0.0)


def summarize_timing_samples(samples: object) -> ResidentTimingSummary:
    """Validate samples and return deterministic min/median/mean/p95 summary.

    The p95 uses the nearest-rank method: ``ceil(0.95 * count) - 1`` in the
    sorted zero-based sequence.
    """
    if not isinstance(samples, tuple):
        raise TypeError("samples must be a tuple.")
    if not samples:
        raise ValueError("samples must not be empty.")
    if len(samples) > MAX_TIMING_SAMPLES:
        raise ValueError("samples exceeds MAX_TIMING_SAMPLES.")
    validated = tuple(
        _require_float(value, "sample", minimum=0.0) for value in samples
    )
    ordered = sorted(validated)
    return ResidentTimingSummary(
        count=len(ordered),
        minimum=ordered[0],
        median=statistics.median(ordered),
        mean=statistics.fmean(ordered),
        p95=ordered[math.ceil(0.95 * len(ordered)) - 1],
    )


@dataclass(frozen=True, slots=True)
class ResidentBenchmarkResult:
    """Validated evidence or explicit non-execution outcome for one case."""

    case_id: str
    timing_mode: str | None
    requested_shape: tuple[int, int, int]
    status: ResidentBenchmarkStatus
    reason: str | None
    samples: tuple[float, ...]
    summary: ResidentTimingSummary | None
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate evidence consistency for the declared result status."""
        if not isinstance(self.case_id, str) or not self.case_id:
            raise TypeError("case_id must be a nonempty string.")
        _validate_shape(self.requested_shape, "requested_shape")
        if not isinstance(self.status, ResidentBenchmarkStatus):
            raise TypeError("status must be a ResidentBenchmarkStatus.")
        if self.timing_mode is not None and not isinstance(
            self.timing_mode, str
        ):
            raise TypeError("timing_mode must be a string or None.")
        if not isinstance(self.samples, tuple):
            raise TypeError("samples must be a tuple.")
        provenance = _freeze_mapping(
            self.provenance, "provenance", nonempty=True
        )
        object.__setattr__(self, "provenance", provenance)
        if self.status is ResidentBenchmarkStatus.EXECUTED:
            if self.timing_mode not in SUPPORTED_TIMING_MODES:
                raise ValueError(
                    "executed results require a supported timing_mode."
                )
            expected = summarize_timing_samples(self.samples)
            if self.summary != expected:
                raise ValueError("summary must exactly match timing samples.")
            if self.reason is not None:
                if not isinstance(self.reason, str):
                    raise TypeError("reason must be a string or None.")
                raise ValueError("executed results must not have a reason.")
        else:
            if (
                self.timing_mode is not None
                or self.samples
                or self.summary is not None
            ):
                raise ValueError(
                    "non-executed results cannot contain timing data."
                )
            if self.reason is not None and not isinstance(self.reason, str):
                raise TypeError("reason must be a string or None.")
            if not isinstance(self.reason, str) or not self.reason:
                raise ValueError(
                    "non-executed results require a nonempty reason."
                )


@dataclass(frozen=True, slots=True)
class ResidentBenchmarkArtifact:
    """Complete immutable resident benchmark metadata, cases, and results."""

    metadata: Mapping[str, Any]
    cases: tuple[ResidentBenchmarkCase, ...]
    results: tuple[ResidentBenchmarkResult, ...]

    def __post_init__(self) -> None:
        """Validate complete metadata and cross-record artifact references."""
        metadata = _validate_metadata(self.metadata)
        object.__setattr__(self, "metadata", metadata)
        if not isinstance(self.cases, tuple) or not all(
            isinstance(case, ResidentBenchmarkCase) for case in self.cases
        ):
            raise TypeError(
                "cases must be a tuple of ResidentBenchmarkCase records."
            )
        if not isinstance(self.results, tuple) or not all(
            isinstance(result, ResidentBenchmarkResult)
            for result in self.results
        ):
            raise TypeError(
                "results must be a tuple of ResidentBenchmarkResult records."
            )
        case_map = {case.case_id: case for case in self.cases}
        if len(case_map) != len(self.cases):
            raise ValueError("cases must have unique case_id values.")
        identities: set[tuple[str, str | None]] = set()
        for result in self.results:
            case = case_map.get(result.case_id)
            if case is None:
                raise ValueError("result references an unknown case_id.")
            if result.requested_shape != case.requested_shape:
                raise ValueError("result requested_shape must match its case.")
            identity = (result.case_id, result.timing_mode)
            if identity in identities:
                raise ValueError(
                    "results must have unique case_id/timing_mode rows."
                )
            identities.add(identity)


def build_resident_benchmark_metadata(
    *,
    timestamp_utc: datetime,
    command: str,
    synchronization_method: str,
    warmup: int,
    timestep_count: int,
    seed: int,
    prepared_signature_digest: str,
    warp_version: Mapping[str, Any],
    device: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Build complete host-only provenance without importing Warp or devices."""
    if not isinstance(timestamp_utc, datetime):
        raise TypeError("timestamp_utc must be a datetime.")
    if (
        timestamp_utc.tzinfo is None
        or timestamp_utc.utcoffset() != timezone.utc.utcoffset(timestamp_utc)
    ):
        raise ValueError("timestamp_utc must be UTC-aware.")
    for name, value in (
        ("command", command),
        ("synchronization_method", synchronization_method),
        ("prepared_signature_digest", prepared_signature_digest),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a nonempty string.")
    metadata = {
        "timestamp_utc": timestamp_utc.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "command": command,
        "python_version": sys.version,
        "platform": platform.platform(),
        "warp_version": _freeze_mapping(warp_version, "warp_version"),
        "device": _freeze_mapping(device, "device"),
        "synchronization_method": synchronization_method,
        "warmup": _require_int(warmup, "warmup"),
        "timestep_count": _require_int(
            timestep_count, "timestep_count", positive=True
        ),
        "seed": _require_int(seed, "seed"),
        "prepared_signature_digest": prepared_signature_digest,
    }
    return _validate_metadata(metadata)


def _normalize_json(value: object) -> Any:
    """Convert supported values into deterministic JSON-compatible values.

    Args:
        value: Value to normalize, including supported frozen records.

    Returns:
        A recursively normalized value containing JSON-compatible types.

    Raises:
        TypeError: If a value or mapping key is unsupported.
        ValueError: If a floating-point value is nonfinite.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Enum):
        return _normalize_json(value.value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite floats cannot be serialized.")
        return value
    if isinstance(value, (tuple, list)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("mapping keys must be strings.")
        return {key: _normalize_json(item) for key, item in value.items()}
    if isinstance(value, ResidentTimingSummary):
        return {
            "count": value.count,
            "minimum": value.minimum,
            "median": value.median,
            "mean": value.mean,
            "p95": value.p95,
        }
    if isinstance(value, ResidentBenchmarkCase):
        return {
            name: _normalize_json(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    if isinstance(value, ResidentBenchmarkResult):
        return {
            name: _normalize_json(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    if isinstance(value, ResidentBenchmarkArtifact):
        return {
            name: _normalize_json(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}.")


def serialize_resident_benchmark_artifact(
    artifact: ResidentBenchmarkArtifact,
) -> str:
    """Serialize a validated artifact in deterministic schema-versioned JSON."""
    if not isinstance(artifact, ResidentBenchmarkArtifact):
        raise TypeError("artifact must be a ResidentBenchmarkArtifact.")
    return (
        json.dumps(
            {
                "schema_version": RESIDENT_BENCHMARK_SCHEMA_VERSION,
                "artifact": _normalize_json(artifact),
            },
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _require_fields(
    value: object, fields: set[str], name: str
) -> Mapping[str, Any]:
    """Require an object to be a dictionary with exactly the given fields.

    Args:
        value: Decoded object to validate.
        fields: Exact set of permitted keys.
        name: Object name used in error messages.

    Returns:
        The validated dictionary.

    Raises:
        ValueError: If the object is not a dictionary with the exact fields.
    """
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} has invalid fields.")
    return value


def deserialize_resident_benchmark_artifact(
    payload: str,
) -> ResidentBenchmarkArtifact:
    """Deserialize a schema envelope through all normal record constructors."""
    if not isinstance(payload, str):
        raise TypeError("payload must be a string.")
    try:
        envelope = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("payload is not valid JSON.") from error
    envelope = _require_fields(
        envelope, {"schema_version", "artifact"}, "envelope"
    )
    if envelope["schema_version"] != RESIDENT_BENCHMARK_SCHEMA_VERSION:
        raise ValueError("unsupported schema_version.")
    raw = _require_fields(
        envelope["artifact"], {"metadata", "cases", "results"}, "artifact"
    )
    cases = []
    for item in raw["cases"]:
        item = _require_fields(
            item, set(ResidentBenchmarkCase.__dataclass_fields__), "case"
        )
        cases.append(
            ResidentBenchmarkCase(
                case_id=item["case_id"],
                requested_shape=tuple(item["requested_shape"]),
                actual_shape=tuple(item["actual_shape"]),
                active_fraction=item["active_fraction"],
                processes=tuple(item["processes"]),
                communication=item["communication"],
                diagnostics=tuple(item["diagnostics"]),
                warmup=item["warmup"],
                timestep_count=item["timestep_count"],
                seed=item["seed"],
            )
        )
    results = []
    for item in raw["results"]:
        item = _require_fields(
            item, set(ResidentBenchmarkResult.__dataclass_fields__), "result"
        )
        summary = item["summary"]
        if summary is not None:
            summary = ResidentTimingSummary(
                **_require_fields(
                    summary,
                    set(ResidentTimingSummary.__dataclass_fields__),
                    "summary",
                )
            )
        results.append(
            ResidentBenchmarkResult(
                case_id=item["case_id"],
                timing_mode=item["timing_mode"],
                requested_shape=tuple(item["requested_shape"]),
                status=ResidentBenchmarkStatus(item["status"]),
                reason=item["reason"],
                samples=tuple(item["samples"]),
                summary=summary,
                provenance=item["provenance"],
            )
        )
    return ResidentBenchmarkArtifact(
        metadata=raw["metadata"], cases=tuple(cases), results=tuple(results)
    )


def write_json_artifact(
    artifact_root: str | os.PathLike[str],
    relative_destination: str | os.PathLike[str],
    payload: object,
) -> Path:
    """Atomically write normalized generic JSON below an existing `.artifacts` root."""
    try:
        serialized = (
            json.dumps(
                _normalize_json(payload),
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
    except (TypeError, ValueError) as error:
        raise OSError("artifact payload serialization failed.") from error
    try:
        root_path = Path(artifact_root)
        if (
            root_path.name != ".artifacts"
            or root_path.is_symlink()
            or not root_path.is_dir()
        ):
            raise OSError(
                "artifact_root must be an existing .artifacts directory."
            )
        root = root_path.resolve(strict=True)
        if root.name != ".artifacts":
            raise OSError("artifact_root must resolve to .artifacts.")
        relative = Path(relative_destination)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative == Path(".")
        ):
            raise OSError(
                "relative_destination must be contained and relative."
            )
        destination = root / relative
        existing = root
        for part in relative.parts:
            candidate = existing / part
            if candidate.exists() or candidate.is_symlink():
                resolved = candidate.resolve(strict=True)
                if resolved != root and root not in resolved.parents:
                    raise OSError("artifact path escapes artifact_root.")
                existing = resolved
            else:
                existing = candidate
        if destination.exists() or destination.is_symlink():
            resolved_destination = destination.resolve(strict=True)
            if (
                resolved_destination != root
                and root not in resolved_destination.parents
            ):
                raise OSError("artifact destination escapes artifact_root.")
        destination.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, TypeError, ValueError) as error:
        raise OSError(f"artifact path validation failed: {error}") from error

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=".resident-benchmark-",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
        return destination
    except OSError as error:
        cleanup_error: OSError | None = None
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except OSError as caught:
                cleanup_error = caught
        if cleanup_error is not None:
            raise OSError(
                "artifact write failed; temporary cleanup also failed."
            ) from cleanup_error
        raise OSError("artifact atomic write failed.") from error
