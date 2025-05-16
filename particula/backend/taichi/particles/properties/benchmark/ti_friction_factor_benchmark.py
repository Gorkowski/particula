"""Benchmarks the Python, Taichi-wrapper, and raw-kernel versions of
get_friction_factor / ti_get_friction_factor / kget_friction_factor."""
# -------- mandatory imports ----------------------------------------------
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

# -------- functions to benchmark -----------------------------------------
from particula.particles.properties.friction_factor_module import (
    get_friction_factor as py_func,
)
from particula.backend.taichi.particles.properties.ti_friction_factor_module import (
    ti_get_friction_factor as ti_func,
    kget_friction_factor   as ti_kernel,
)

# -------- reproducibility -------------------------------------------------
RNG_SEED      = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)     # 10² … 10⁸
ti.init(arch=ti.cpu)


def benchmark_friction_factor_csv() -> None:
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
        radius = rng.random(n, dtype=np.float64) * 3.0e-7 + 1.0e-8  # 10–300 nm
        corr   = rng.random(n, dtype=np.float64) * 0.5     + 1.0    # 1.0–1.5
        mu     = np.float64(1.8e-5)                                  # Pa·s (scalar)

        # ----- Taichi buffers (create once per length) -----------------
        radius_ti = ti.ndarray(dtype=ti.f64, shape=n)
        corr_ti   = ti.ndarray(dtype=ti.f64, shape=n)
        res_ti    = ti.ndarray(dtype=ti.f64, shape=n)
        radius_ti.from_numpy(radius)
        corr_ti.from_numpy(corr)

        # ----- timing --------------------------------------------------
        stats_py     = get_function_benchmark(
            lambda: py_func(radius, mu, corr), ops_per_call=n
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_func(radius, mu, corr), ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: ti_kernel(radius_ti, mu, corr_ti, res_ti), ops_per_call=n
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
    header      = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    # ------------------------ output directory -------------------------
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------ CSV --------------------------------------
    csv_path = os.path.join(out_dir, "ti_friction_factor_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    # ------------------------ system info JSON -------------------------
    with open(os.path.join(out_dir, "system_info.json"),
              "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    # ------------------------ throughput plot --------------------------
    plot_throughput_vs_array_length(
        header,
        rows,
        "Friction-factor throughput benchmark",
        os.path.join(out_dir, "ti_friction_factor_benchmark.png"),
    )


if __name__ == "__main__":
    benchmark_friction_factor_csv()
