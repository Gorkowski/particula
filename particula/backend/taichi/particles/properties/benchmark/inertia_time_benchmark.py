# 1. --- imports ------------------------------------------------------------
"""Benchmarks Python, Taichi wrapper, and raw kernel versions of
get_particle_inertia_time."""
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
from particula.particles.properties.inertia_time import (
    get_particle_inertia_time as py_func
)
from particula.backend.taichi.particles.properties.ti_inertia_time_module import (
    ti_get_particle_inertia_time as ti_func,
    kget_particle_inertia_time  as ti_kernel,
)

# 2. --- benchmark configuration -------------------------------------------
RNG_SEED      = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)     # 10² … 10⁸
ti.init(arch=ti.cpu)


# 3. --- main benchmark routine --------------------------------------------
def benchmark_particle_inertia_time_csv() -> None:
    """Time pure-Python, Taichi wrapper, and raw kernel, save CSV/JSON/PNG."""
    rows: list[list[float]] = []
    rng  = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        # ---------- random input data -------------------------------------
        r      = rng.random(n, dtype=np.float64) * 5e-6 + 1e-9      # m
        rho_p  = rng.random(n, dtype=np.float64) * 400.0 + 800.0    # kg/m³
        rho_f  = np.full(n, 1.2,     dtype=np.float64)              # kg/m³
        nu     = np.full(n, 1.5e-5,  dtype=np.float64)              # m²/s

        # ---------- Taichi buffers ----------------------------------------
        r_ti, rho_p_ti, rho_f_ti, nu_ti, res_ti = [
            ti.ndarray(dtype=ti.f64, shape=n) for _ in range(5)
        ]
        r_ti.from_numpy(r); rho_p_ti.from_numpy(rho_p)
        rho_f_ti.from_numpy(rho_f); nu_ti.from_numpy(nu)

        # ---------- timing ------------------------------------------------
        stats_py     = get_function_benchmark(
            lambda: py_func(r, rho_p, rho_f, nu), ops_per_call=n
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_func(r, rho_p, rho_f, nu), ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: ti_kernel(r_ti, rho_p_ti, rho_f_ti, nu_ti, res_ti),
            ops_per_call=n,
        )

        # ---------- collect CSV row ---------------------------------------
        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    # ---------- header ----------------------------------------------------
    python_hdr = ["python_"         + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_"         + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_"  + h for h in stats_kernel["array_headers"]]
    header     = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    # ---------- outputs ---------------------------------------------------
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "inertia_time_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    with open(os.path.join(out_dir, "system_info.json"),
              "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        rows,
        "Particle inertia time throughput benchmark",
        os.path.join(out_dir, "inertia_time_benchmark.png"),
    )


# 4. --- entry-point guard --------------------------------------------------
if __name__ == "__main__":
    benchmark_particle_inertia_time_csv()
