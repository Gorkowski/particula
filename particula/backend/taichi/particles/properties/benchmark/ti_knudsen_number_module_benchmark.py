"""
Benchmarks the reference Python, the Taichi wrapper, and the raw Taichi kernel.
"""

import os
import numpy as np
import taichi as ti
import json
from particula.backend.benchmark import (
    get_function_benchmark,
    get_system_info,
    save_combined_csv,
    plot_throughput_vs_array_length,
)
from particula.particles.properties.knudsen_number_module import (
    get_knudsen_number as get_knudsen_number_python,
)
from particula.backend.taichi.particles.properties.ti_knudsen_number_module import (
    ti_get_knudsen_number,
    kget_knudsen_number,
)

ti.init(arch=ti.cpu)


def benchmark_ti_knudsen_number_module_csv():
    """
    Time pure-Python, Taichi wrapper, and raw kernel over ARRAY_LENGTHS,
    then save CSV, JSON, and PNG into ./benchmark_outputs/.
    """
    rows: list[list[float]] = []
    rng = np.random.default_rng(seed=42)
    array_lengths = np.logspace(2, 8, 10, dtype=int)

    # ------------------------------------------------------------------ #
    #  LOOP OVER ARRAY LENGTHS – *no separate helper function*           #
    # ------------------------------------------------------------------ #
    for n in array_lengths:
        mfp = rng.random(n, dtype=np.float64) + 1e-9
        pr  = rng.random(n, dtype=np.float64) + 1e-9

        # Taichi buffers
        mfp_ti = ti.ndarray(dtype=ti.f64, shape=n)
        pr_ti  = ti.ndarray(dtype=ti.f64, shape=n)
        res_ti = ti.ndarray(dtype=ti.f64, shape=n)
        mfp_ti.from_numpy(mfp)
        pr_ti.from_numpy(pr)

        stats_py     = get_function_benchmark(
            lambda: get_knudsen_number_python(mfp, pr), ops_per_call=n
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_get_knudsen_number(mfp, pr),    ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: kget_knudsen_number(mfp_ti, pr_ti, res_ti), ops_per_call=n
        )

        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    # ---------- header --------------------------------------------------
    python_hdr  = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr  = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr  = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    # sub-folder relative to this test file
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "ti_knudsen_number_module_benchmark.csv")
    save_combined_csv(csv_path, header, rows)  # accept future matrices

    # Save system information for reproducibility
    sysinfo_path = os.path.join(out_dir, "system_info.json")
    with open(sysinfo_path, "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    # create PNG with generic helper
    png_path = os.path.join(out_dir, "ti_knudsen_number_module_benchmark.png")
    plot_throughput_vs_array_length(
        header,
        rows,
        "Knudsen-number throughput benchmark",
        png_path,
    )


if __name__ == "__main__":
    benchmark_ti_knudsen_number_module_csv()
