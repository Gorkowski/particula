"""
Benchmarks the reference Python and Taichi implementations of
`get_knudsen_number` for 10 array lengths between 10 and 10 000 elements
and stores the timing results in a CSV file inside the test folder.
"""

from __future__ import annotations
import os
import csv
import numpy as np
import taichi as ti
import particula as par
from particula.backend.benchmark import get_function_benchmark
from particula.particles.properties.knudsen_number_module import (
    get_knudsen_number as get_knudsen_number_python,
)
from particula.backend.taichi.particles.properties.ti_knudsen_number_module import (
    get_knudsen_number_taichi,
)

# Initialise Taichi only if not yet done
try:
    ti.init(arch=ti.cpu)
except RuntimeError:
    pass  # already initialised elsewhere


def _run_benchmarks() -> list[tuple[int, float, float]]:
    """Return list of (n, mean_time_py, mean_time_ti)."""
    rng = np.random.default_rng(seed=42)
    lengths = np.logspace(np.log10(10), np.log10(10_000), 10, dtype=int)
    rows: list[tuple[int, float, float]] = []

    for n in lengths:
        mfp = rng.random(n, dtype=np.float64) + 1e-9
        pr = rng.random(n, dtype=np.float64) + 1e-9

        # zero-arg lambdas for the benchmark helper
        py_call = lambda: get_knudsen_number_python(mfp, pr)
        ti_call = lambda: get_knudsen_number_taichi(mfp, pr)

        stats_py = get_function_benchmark(py_call, ops_per_call=n, repeats=1)
        stats_ti = get_function_benchmark(ti_call, ops_per_call=n, repeats=1)

        rows.append((n, stats_py["mean_time_s"], stats_ti["mean_time_s"]))

    return rows


def test_knudsen_benchmark_creates_csv():
    """Benchmark both versions and write CSV into ./benchmark_outputs/."""
    results = _run_benchmarks()

    # sub-folder relative to this test file
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "knudsen_benchmark.csv")

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["array_length", "python_mean_time_s", "taichi_mean_time_s"])
        writer.writerows(results)

    # minimal assertion – file exists and contains 11 lines (header + 10 rows)
    assert os.path.isfile(csv_path)
    with open(csv_path, "r", newline="") as fh:
        assert sum(1 for _ in fh) == 11
