"""Benchmarks the reference Python, Taichi wrapper, and raw Taichi kernel for
particle Reynolds number (Reₚ = 2·r·v / ν)."""

# ---------- std/third-party imports ---------------------------------
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

# ---------- functions to benchmark ---------------------------------
from particula.particles.properties.reynolds_number import (
    get_particle_reynolds_number as py_func,
)
from particula.backend.taichi.particles.properties.ti_reynolds_number_module import (
    ti_get_particle_reynolds_number as ti_func,
    kget_particle_reynolds_number  as ti_kernel,
)

# ---------- reproducibility / Taichi backend ------------------------
RNG_SEED      = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)
ti.init(arch=ti.cpu)

# ---------- main benchmark routine ---------------------------------
def benchmark_reynolds_number_csv():
    """
    Time pure-Python, Taichi wrapper, and raw kernel over ARRAY_LENGTHS,
    then save CSV, JSON, and PNG into ./benchmark_outputs/.
    """
    rows, rng = [], np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        pr = rng.random(n, dtype=np.float64) * 1e-6 + 1e-12
        pv = rng.random(n, dtype=np.float64) * 1e-1 + 1e-12
        kv = rng.random(n, dtype=np.float64) * 1e-5 + 1e-12

        # Taichi buffers
        pr_ti, pv_ti, kv_ti = (
            ti.ndarray(dtype=ti.f64, shape=n) for _ in range(3)
        )
        res_ti = ti.ndarray(dtype=ti.f64, shape=n)
        pr_ti.from_numpy(pr);  pv_ti.from_numpy(pv);  kv_ti.from_numpy(kv)

        # timing
        stats_py     = get_function_benchmark(
            lambda: py_func(pr, pv, kv), ops_per_call=n
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_func(pr, pv, kv), ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: ti_kernel(pr_ti, pv_ti, kv_ti, res_ti), ops_per_call=n
        )

        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    # header construction
    python_hdr = ["python_" + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_" + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header     = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    # output directory
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    # CSV
    save_combined_csv(
        os.path.join(out_dir, "reynolds_number_benchmark.csv"),
        header, rows
    )

    # system info JSON
    with open(os.path.join(out_dir, "system_info.json"), "w",
              encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    # throughput plot
    plot_throughput_vs_array_length(
        header,
        rows,
        "Particle Reynolds number throughput benchmark",
        os.path.join(out_dir, "reynolds_number_benchmark.png"),
    )

# ---------- entry-point guard --------------------------------------
if __name__ == "__main__":
    benchmark_reynolds_number_csv()
