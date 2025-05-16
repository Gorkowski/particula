"""
Benchmarks the reference Python, the Taichi wrapper, and the raw Taichi kernel implementations of
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
    kget_knudsen_number,
)

import matplotlib.pyplot as plt           # NEW

ti.init(arch=ti.cpu)


def _run_benchmarks() -> tuple[list, list]:
    """Return list of rows: [n, *python_stats..., *taichi_stats...]."""
    rng = np.random.default_rng(seed=42)
    lengths = np.logspace(2, 8, 10, dtype=int)
    rows: list = []

    for n in lengths:
        mfp = rng.random(n, dtype=np.float64) + 1e-9
        pr = rng.random(n, dtype=np.float64) + 1e-9

        # Build Taichi NDArrays once per array length
        mfp_ti = ti.ndarray(dtype=ti.f64, shape=mfp.shape)
        pr_ti = ti.ndarray(dtype=ti.f64, shape=pr.shape)
        res_ti = ti.ndarray(dtype=ti.f64, shape=mfp.shape)
        mfp_ti.from_numpy(mfp)
        pr_ti.from_numpy(pr)

        # zero-arg lambdas for the benchmark helper
        py_call = lambda: get_knudsen_number_python(mfp, pr)
        ti_call = lambda: ti_get_knudsen_number(mfp, pr)
        ti_kernel_call = lambda: kget_knudsen_number(mfp_ti, pr_ti, res_ti)

        stats_py = get_function_benchmark(py_call, ops_per_call=len(mfp))
        stats_ti = get_function_benchmark(ti_call, ops_per_call=len(mfp))
        stats_kernel = get_function_benchmark(
            ti_kernel_call, ops_per_call=len(mfp)
        )

        row = [
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ]
        rows.append(row)
    python_header = ["python_" + k for k in stats_py["array_headers"]]
    taichi_header = ["taichi_" + k for k in stats_ti["array_headers"]]
    taichi_kernel_header = [
        "taichi_kernel_" + k for k in stats_kernel["array_headers"]
    ]
    header = (
        ["array_length"] + python_header + taichi_header + taichi_kernel_header
    )
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

    # ---------- plot throughput vs array_length -----------------------------
    # column indices for the three throughput fields
    idx_py      = header.index("python_throughput_calls_per_s")
    idx_ti      = header.index("taichi_throughput_calls_per_s")
    idx_kernel  = header.index("taichi_kernel_throughput_calls_per_s")

    array_lengths   = [row[0]           for row in rows]
    throughput_py   = [row[idx_py]      for row in rows]
    throughput_ti   = [row[idx_ti]      for row in rows]
    throughput_kern = [row[idx_kernel]  for row in rows]

    plt.figure()
    plt.loglog(array_lengths, throughput_py,   "o-", label="Python")
    plt.loglog(array_lengths, throughput_ti,   "s-", label="Taichi wrapper")
    plt.loglog(array_lengths, throughput_kern, "^-", label="Taichi kernel")
    plt.xlabel("Array length")
    plt.ylabel("Throughput (calls/s)")
    plt.title("Knudsen-number throughput benchmark")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)

    png_path = os.path.join(out_dir, "knudsen_benchmark_throughput.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    knudsen_benchmark_csv()
