"""
Benchmarks the reference Python and Taichi implementations of
`get_knudsen_number` for 10 array lengths between 10 and 10 000 elements
and stores the timing results in a CSV file inside the test folder.
"""

import os
import numpy as np
import taichi as ti
import json
from particula.backend.benchmark import (
    get_function_benchmark,
    get_system_info,
    save_combined_csv,
)
from particula.particles.properties.knudsen_number_module import (
    get_knudsen_number as get_knudsen_number_python,
)
from particula.backend.taichi.particles.properties.ti_knudsen_number_module import (
    ti_get_knudsen_number,
)

ti.init(arch=ti.cpu)


def _run_benchmarks() -> list[list[float]]:
    """Return list of rows: [n, *python_stats..., *taichi_stats...]."""
    rng = np.random.default_rng(seed=42)
    lengths = np.logspace(2, 6, 10, dtype=int)
    rows: list = []

    for n in lengths:
        mfp = rng.random(n, dtype=np.float64) + 1e-9
        pr = rng.random(n, dtype=np.float64) + 1e-9

        # zero-arg lambdas for the benchmark helper
        py_call = lambda: get_knudsen_number_python(mfp, pr)
        ti_call = lambda: ti_get_knudsen_number(mfp, pr)

        stats_py = get_function_benchmark(py_call, ops_per_call=len(mfp))
        stats_ti = get_function_benchmark(ti_call, ops_per_call=len(mfp))

        row = [n, *stats_py["array_stats"], *stats_ti["array_stats"]]
        rows.append(row)
    python_header = ["python_" + k for k in stats_py["array_headers"]]
    taichi_header = ["taichi_" + k for k in stats_ti["array_headers"]]
    header = ["array_length"] + python_header + taichi_header
    return rows, header


def knudsen_benchmark_csv():
    """Benchmark both versions and write CSV into ./benchmark_outputs/."""
    rows, header = _run_benchmarks()

    # sub-folder relative to this test file
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "knudsen_benchmark.csv")
    save_combined_csv(csv_path, header, rows)  # accept future matrices

    # Save system information for reproducibility
    sysinfo_path = os.path.join(out_dir, "system_info.json")
    with open(sysinfo_path, "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)


if __name__ == "__main__":
    knudsen_benchmark_csv()
