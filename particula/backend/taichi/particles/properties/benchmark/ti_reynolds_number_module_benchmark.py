"""Benchmarks the Python, Taichi-wrapper and raw-kernel versions of the
particle-Reynolds-number routine."""
# ---------- imports ---------------------------------------------------------
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

# functions to benchmark
from particula.particles.properties.reynolds_number_module import (
    get_particle_reynolds_number as py_func,
)
from particula.backend.taichi.particles.properties.ti_reynolds_number_module import (
    ti_get_particle_reynolds_number as ti_func,
    kget_particle_reynolds_number   as ti_kernel,
)

# ---------- fixed config ----------------------------------------------------
RNG_SEED      = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)   # 10² … 10⁸
ti.init(arch=ti.cpu)

# ---------- main benchmark --------------------------------------------------
def benchmark_particle_reynolds_number_csv() -> None:
    """
    Time pure-Python, Taichi wrapper, and raw kernel over ARRAY_LENGTHS,
    then save CSV, JSON, and PNG into ./benchmark_outputs/.
    """
    rows: list[list[float]] = []
    rng  = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        # ---------------- random input -------------------------------------
        pr = rng.random(n, dtype=np.float64) + 1e-9   # particle radius
        pv = rng.random(n, dtype=np.float64) + 1e-9   # particle velocity
        kv = rng.random(n, dtype=np.float64) + 1e-9   # kinematic viscosity

        # ---------------- Taichi buffers -----------------------------------
        pr_ti = ti.ndarray(dtype=ti.f64, shape=n)
        pv_ti = ti.ndarray(dtype=ti.f64, shape=n)
        kv_ti = ti.ndarray(dtype=ti.f64, shape=n)
        out_ti = ti.ndarray(dtype=ti.f64, shape=n)
        pr_ti.from_numpy(pr)
        pv_ti.from_numpy(pv)
        kv_ti.from_numpy(kv)

        # ---------------- timing -------------------------------------------
        stats_py     = get_function_benchmark(
            lambda: py_func(pr, pv, kv), ops_per_call=n
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_func(pr, pv, kv), ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: ti_kernel(pr_ti, pv_ti, kv_ti, out_ti), ops_per_call=n
        )

        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    # -------------- header construction ------------------------------------
    python_hdr = ["python_"         + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_"         + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_"  + h for h in stats_kernel["array_headers"]]
    header     = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    # -------------- output paths -------------------------------------------
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "particle_reynolds_number_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        rows,
        "Particle Reynolds-number throughput benchmark",
        os.path.join(out_dir, "particle_reynolds_number_benchmark.png"),
    )

# ---------- entry-point -----------------------------------------------------
if __name__ == "__main__":
    benchmark_particle_reynolds_number_csv()
