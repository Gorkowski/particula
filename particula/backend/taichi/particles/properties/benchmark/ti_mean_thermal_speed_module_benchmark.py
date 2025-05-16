"""Benchmarks the reference Python, Taichi wrapper, and raw kernel."""
import os
import json
import numpy as np
import taichi as ti
from particula.backend.benchmark import (
    get_function_benchmark,
    get_system_info,
    save_combined_csv,
    plot_throughput_vs_array_length,
)

# ─ functions to benchmark ───────────────────────────────────────────────
from particula.particles.properties.mean_thermal_speed_module import (
    get_mean_thermal_speed as py_func,
)
from particula.backend.taichi.particles.properties.ti_mean_thermal_speed_module import (
    ti_get_mean_thermal_speed as ti_func,
    kget_mean_thermal_speed   as ti_kernel,
)

RNG_SEED = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)   # 10² … 10⁸
ti.init(arch=ti.cpu)                                # CPU-only benchmark

def benchmark_ti_mean_thermal_speed_module_csv():
    """
    Time pure-Python, Taichi wrapper, and raw kernel over ARRAY_LENGTHS,
    then save CSV, JSON, and PNG into ./benchmark_outputs/.
    """
    rows = []
    rng  = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        # ─────────── random positive inputs ───────────
        pm   = rng.random(n, dtype=np.float64) * 2.0e-20 + 1.0e-21   # kg
        temp = rng.random(n, dtype=np.float64) * 100.0    + 250.0     # K

        # ─────────── Taichi buffers ───────────
        pm_ti   = ti.ndarray(dtype=ti.f64, shape=n)
        temp_ti = ti.ndarray(dtype=ti.f64, shape=n)
        res_ti  = ti.ndarray(dtype=ti.f64, shape=n)
        pm_ti.from_numpy(pm)
        temp_ti.from_numpy(temp)

        # ─────────── timing ───────────
        stats_py     = get_function_benchmark(
            lambda: py_func(pm, temp), ops_per_call=n
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_func(pm, temp), ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: ti_kernel(pm_ti, temp_ti, res_ti), ops_per_call=n
        )

        # ─────────── collect one CSV row ───────────
        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    # ─────────── header construction ───────────
    python_hdr  = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr  = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr  = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    # ─────────── output paths ───────────
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "ti_mean_thermal_speed_module_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        rows,
        "Mean thermal speed throughput benchmark",
        os.path.join(out_dir, "ti_mean_thermal_speed_module_benchmark.png"),
    )

if __name__ == "__main__":
    benchmark_ti_mean_thermal_speed_module_csv()
