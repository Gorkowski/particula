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
import json
from particula.backend.benchmark import get_function_benchmark, get_system_info
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

def _save_combined_csv(csv_path: str, header: list[str], *result_matrices):
    """
    Write one or more 2-D `result_matrices` into a single CSV file.

    Each matrix is an iterable of rows; every row length must match
    `header`.  Matrices are concatenated in the order provided.
    """
    rows = [row for matrix in result_matrices for row in matrix]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)

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

    header = ["array_length", "python_mean_time_s", "taichi_mean_time_s"]
    _save_combined_csv(csv_path, header, results)   # accept future matrices

    # Save system information for reproducibility
    sysinfo_path = os.path.join(out_dir, "system_info.json")
    with open(sysinfo_path, "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    assert os.path.isfile(csv_path)
    assert os.path.isfile(sysinfo_path)

    # CSV should contain header + number-of-rows lines
    with open(csv_path, "r", newline="") as fh:
        assert sum(1 for _ in fh) == len(results) + 1
