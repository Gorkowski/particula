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
from particula.particles.properties.partial_pressure_module import (
    get_partial_pressure_delta as py_func,
)
from particula.backend.taichi.particles.properties.ti_partial_pressure_module import (
    ti_get_partial_pressure_delta as ti_func,
    kget_partial_pressure_delta as ti_kernel,
)

# -- fixed RNG and Taichi backend for reproducibility -------------------
RNG_SEED = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)   # 10² … 10⁸
ti.init(arch=ti.cpu)

def benchmark_partial_pressure_delta_csv():
    """Time Python, Taichi wrapper, and raw kernel over ARRAY_LENGTHS, then save CSV, JSON, and PNG."""
    rows = []
    rng = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        # ----- random input data ---------------------------------------
        pg = rng.random(n, dtype=np.float64) + 1e-9
        pp = rng.random(n, dtype=np.float64) + 1e-9
        kt = rng.random(n, dtype=np.float64) + 1e-9

        # ----- Taichi buffers (create once per length) -----------------
        pg_ti = ti.ndarray(dtype=ti.f64, shape=n)
        pp_ti = ti.ndarray(dtype=ti.f64, shape=n)
        kt_ti = ti.ndarray(dtype=ti.f64, shape=n)
        res_ti = ti.ndarray(dtype=ti.f64, shape=n)
        pg_ti.from_numpy(pg)
        pp_ti.from_numpy(pp)
        kt_ti.from_numpy(kt)

        # ----- timing --------------------------------------------------
        stats_py = get_function_benchmark(
            lambda: py_func(pg, pp, kt), ops_per_call=n
        )
        stats_ti = get_function_benchmark(
            lambda: ti_func(pg, pp, kt), ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: ti_kernel(pg_ti, pp_ti, kt_ti, res_ti), ops_per_call=n
        )

        # ----- collect one CSV row ------------------------------------
        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    # ------------------------ header construction ----------------------
    python_hdr = ["python_" + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_" + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    # ------------------------ output directory -------------------------
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------ CSV --------------------------------------
    csv_path = os.path.join(out_dir, "partial_pressure_delta_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    # ------------------------ system info JSON -------------------------
    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    # ------------------------ throughput plot --------------------------
    plot_throughput_vs_array_length(
        header,
        rows,
        "Partial Pressure Delta throughput benchmark",
        os.path.join(out_dir, "partial_pressure_delta_benchmark.png"),
    )

if __name__ == "__main__":
    benchmark_partial_pressure_delta_csv()
