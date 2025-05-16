"""Benchmarks the reference Python, Taichi wrapper, and raw Taichi kernel."""
# ── imports ────────────────────────────────────────────────────────────────
import os
import json
import numpy as np
import taichi as ti
from particula.backend.benchmark import (                       # utilities
    get_function_benchmark,
    get_system_info,
    save_combined_csv,
    plot_throughput_vs_array_length,
)

# functions to benchmark
from particula.gas.properties.concentration_function import (
    get_concentration_from_pressure as py_func,
)
from particula.backend.taichi.gas.properties.ti_concentration_from_pressure_module import (
    ti_get_concentration_from_pressure as ti_func,
    kget_concentration_from_pressure as ti_kernel,
)

# ── benchmark configuration ────────────────────────────────────────────────
RNG_SEED = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)        # 10² … 10⁸
ti.init(arch=ti.cpu)


def benchmark_concentration_from_pressure_csv() -> None:
    """
    Time pure-Python, Taichi wrapper, and raw kernel over ARRAY_LENGTHS,
    then save CSV, JSON, and PNG into ./benchmark_outputs/.
    """
    rows: list[list[float]] = []
    rng = np.random.default_rng(seed=RNG_SEED)

    # ── loop over array lengths ────────────────────────────────────────────
    for n in ARRAY_LENGTHS:
        # random input data (physically sensible ranges)
        pp = rng.random(n, dtype=np.float64) * 1.0e5 + 1.0      # Pa
        mm = rng.random(n, dtype=np.float64) * 0.04   + 0.002   # kg mol⁻¹
        tt = rng.random(n, dtype=np.float64) * 300.0  + 200.0   # K

        # Taichi buffers
        pp_ti = ti.ndarray(dtype=ti.f64, shape=n)
        mm_ti = ti.ndarray(dtype=ti.f64, shape=n)
        tt_ti = ti.ndarray(dtype=ti.f64, shape=n)
        res_ti = ti.ndarray(dtype=ti.f64, shape=n)
        pp_ti.from_numpy(pp)
        mm_ti.from_numpy(mm)
        tt_ti.from_numpy(tt)

        # timing
        stats_py = get_function_benchmark(
            lambda: py_func(pp, mm, tt), ops_per_call=n
        )
        stats_ti = get_function_benchmark(
            lambda: ti_func(pp, mm, tt), ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: ti_kernel(pp_ti, mm_ti, tt_ti, res_ti), ops_per_call=n
        )

        # collect CSV row
        rows.append(
            [n, *stats_py["array_stats"], *stats_ti["array_stats"], *stats_kernel["array_stats"]]
        )

    # ── header construction ────────────────────────────────────────────────
    python_hdr = ["python_" + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_" + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    # ── output directory ───────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    # ── CSV ────────────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "concentration_from_pressure_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    # ── system-info JSON ───────────────────────────────────────────────────
    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    # ── throughput plot ────────────────────────────────────────────────────
    plot_throughput_vs_array_length(
        header,
        rows,
        "Concentration-from-pressure throughput benchmark",
        os.path.join(out_dir, "concentration_from_pressure_benchmark.png"),
    )


if __name__ == "__main__":        # entry-point guard
    benchmark_concentration_from_pressure_csv()
