"""
Benchmarks the reference Python, Taichi wrapper, and raw Taichi kernel for
get_aerodynamic_mobility().
"""
# ---------------------------- imports ------------------------------------
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

# functions to benchmark --------------------------------------------------
from particula.particles.properties.aerodynamic_mobility_module import (
    get_aerodynamic_mobility as py_func,
)
from particula.backend.taichi.particles.properties.ti_aerodynamic_mobility_module import (
    ti_get_aerodynamic_mobility as ti_func,                               # Taichi wrapper
    kget_aerodynamic_mobility  as ti_kernel,                              # raw kernel
)

# ----------------------- benchmark configuration -------------------------
RNG_SEED      = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)   # 10² … 10⁸
ti.init(arch=ti.cpu)


def benchmark_aerodynamic_mobility_csv() -> None:
    """
    Time pure-Python, Taichi wrapper, and raw kernel over ARRAY_LENGTHS,
    then save CSV, JSON, and PNG into ./benchmark_outputs/.
    """
    rows: list[list[float]] = []
    rng  = np.random.default_rng(seed=RNG_SEED)

    # ------------------------------------------------------------------ #
    #  LOOP OVER ARRAY LENGTHS – *no separate helper function*           #
    # ------------------------------------------------------------------ #
    for n in ARRAY_LENGTHS:
        # ----- random input data ---------------------------------------
        r  = rng.random(n, dtype=np.float64) * 9e-7 + 1e-8        # particle radius
        c  = 1.0 + rng.random(n, dtype=np.float64) * 0.5          # slip-corr. factor
        mu = rng.random(n, dtype=np.float64) * 9e-6 + 1e-6        # dynamic viscosity

        # ----- Taichi buffers -----------------------------------------
        r_ti  = ti.ndarray(dtype=ti.f64, shape=n)
        c_ti  = ti.ndarray(dtype=ti.f64, shape=n)
        mu_ti = ti.ndarray(dtype=ti.f64, shape=n)
        res_ti = ti.ndarray(dtype=ti.f64, shape=n)
        r_ti.from_numpy(r); c_ti.from_numpy(c); mu_ti.from_numpy(mu)

        # ----- timing --------------------------------------------------
        stats_py = get_function_benchmark(
            lambda: py_func(r, c, mu), ops_per_call=n
        )
        stats_ti = get_function_benchmark(
            lambda: ti_func(r, c, mu), ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: ti_kernel(r_ti, c_ti, mu_ti, res_ti), ops_per_call=n
        )

        # ----- collect one CSV row ------------------------------------
        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    # ------------------------ header construction ----------------------
    python_hdr  = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr  = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr  = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    # ------------------------ output directory -------------------------
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------ CSV --------------------------------------
    csv_path = os.path.join(out_dir, "aerodynamic_mobility_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    # ------------------------ system info JSON -------------------------
    with open(os.path.join(out_dir, "system_info.json"),
              "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    # ------------------------ throughput plot --------------------------
    plot_throughput_vs_array_length(
        header,
        rows,
        "Aerodynamic mobility throughput benchmark",
        os.path.join(out_dir, "aerodynamic_mobility_benchmark.png"),
    )


if __name__ == "__main__":
    benchmark_aerodynamic_mobility_csv()
