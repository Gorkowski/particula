# --- standard & required imports -----------------------------------------
import os, json
import numpy as np
import taichi as ti
from particula.backend.benchmark import (
    get_function_benchmark, get_system_info,
    save_combined_csv, plot_throughput_vs_array_length,
)

# --- functions to benchmark ----------------------------------------------
from particula.gas.properties.vapor_pressure_module import (
    get_antoine_vapor_pressure       as py_antoine,
    get_clausius_clapeyron_vapor_pressure as py_cc,
    get_buck_vapor_pressure          as py_buck,
)
from particula.backend.taichi.gas.properties.ti_vapor_pressure_module import (
    ti_get_antoine_vapor_pressure    as ti_antoine,
    ti_get_clausius_clapeyron_vapor_pressure as ti_cc,
    ti_get_buck_vapor_pressure       as ti_buck,
    kget_antoine_vapor_pressure,
    kget_clausius_clapeyron_vapor_pressure,
    kget_buck_vapor_pressure,
)

# --- reproducibility ------------------------------------------------------
RNG_SEED      = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)     # 10² … 10⁸
ti.init(arch=ti.cpu)
R_GAS = 8.31446261815324                              # CC-kernel arg

# -------------------------------------------------------------------------
# Benchmark #1 – Antoine vapour pressure
# -------------------------------------------------------------------------
def benchmark_antoine_vapor_pressure_csv():
    rows, rng = [], np.random.default_rng(seed=RNG_SEED)
    a_val, b_val, c_val = 8.07131, 1730.63, 233.426   # fixed water params
    for n in ARRAY_LENGTHS:
        # ------------ input ------------------------------------------------
        T      = rng.random(n, dtype=np.float64) * 200.0 + 250.0  # 250–450 K
        a, b, c = np.full(n, a_val), np.full(n, b_val), np.full(n, c_val)
        # ------------ Ti buffers -------------------------------------------
        a_ti = ti.ndarray(dtype=ti.f64, shape=n); a_ti.from_numpy(a)
        b_ti = ti.ndarray(dtype=ti.f64, shape=n); b_ti.from_numpy(b)
        c_ti = ti.ndarray(dtype=ti.f64, shape=n); c_ti.from_numpy(c)
        T_ti = ti.ndarray(dtype=ti.f64, shape=n);  T_ti.from_numpy(T)
        res  = ti.ndarray(dtype=ti.f64, shape=n)
        # ------------ timing -----------------------------------------------
        stats_py  = get_function_benchmark(
            lambda: py_antoine(a_val, b_val, c_val, T), ops_per_call=n
        )
        stats_ti  = get_function_benchmark(
            lambda: ti_antoine(a_val, b_val, c_val, T), ops_per_call=n
        )
        stats_k   = get_function_benchmark(
            lambda: kget_antoine_vapor_pressure(a_ti, b_ti, c_ti, T_ti, res),
            ops_per_call=n
        )
        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_k["array_stats"],
        ])
    # ------------------------------ IO + plotting --------------------------
    python_hdr = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in stats_k["array_headers"]]
    header     = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)
    save_combined_csv(os.path.join(out_dir, "antoine_vapor_pressure_benchmark.csv"),
                      header, rows)
    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)
    plot_throughput_vs_array_length(
        header, rows, "Antoine vapor-pressure throughput benchmark",
        os.path.join(out_dir, "antoine_vapor_pressure_benchmark.png"),
    )

# -------------------------------------------------------------------------
# Benchmark #2 – Clausius-Clapeyron vapour pressure
# -------------------------------------------------------------------------
def benchmark_clausius_clapeyron_vapor_pressure_csv():
    rows, rng = [], np.random.default_rng(seed=RNG_SEED)
    lh_val, T0_val, P0_val = 4.066e4, 373.15, 1.01325e5
    for n in ARRAY_LENGTHS:
        T   = rng.random(n, dtype=np.float64) * 200.0 + 250.0
        lh, T0, P0 = (np.full(n, lh_val), np.full(n, T0_val), np.full(n, P0_val))
        lh_ti = ti.ndarray(dtype=ti.f64, shape=n); lh_ti.from_numpy(lh)
        T0_ti = ti.ndarray(dtype=ti.f64, shape=n); T0_ti.from_numpy(T0)
        P0_ti = ti.ndarray(dtype=ti.f64, shape=n); P0_ti.from_numpy(P0)
        T_ti  = ti.ndarray(dtype=ti.f64, shape=n); T_ti.from_numpy(T)
        res   = ti.ndarray(dtype=ti.f64, shape=n)

        stats_py = get_function_benchmark(
            lambda: py_cc(lh_val, T0_val, P0_val, T), ops_per_call=n
        )
        stats_ti = get_function_benchmark(
            lambda: ti_cc(lh_val, T0_val, P0_val, T), ops_per_call=n
        )
        stats_k  = get_function_benchmark(
            lambda: kget_clausius_clapeyron_vapor_pressure(
                lh_ti, T0_ti, P0_ti, T_ti, R_GAS, res
            ),
            ops_per_call=n,
        )
        rows.append([n, *stats_py["array_stats"], *stats_ti["array_stats"],
                     *stats_k["array_stats"]])

    python_hdr = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in stats_k["array_headers"]]
    header     = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)
    save_combined_csv(os.path.join(out_dir, "clausius_clapeyron_benchmark.csv"),
                      header, rows)
    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)
    plot_throughput_vs_array_length(
        header, rows, "Clausius-Clapeyron throughput benchmark",
        os.path.join(out_dir, "clausius_clapeyron_benchmark.png"),
    )

# -------------------------------------------------------------------------
# Benchmark #3 – Buck vapour pressure
# -------------------------------------------------------------------------
def benchmark_buck_vapor_pressure_csv():
    rows, rng = [], np.random.default_rng(seed=RNG_SEED)
    for n in ARRAY_LENGTHS:
        T = rng.random(n, dtype=np.float64) * 200.0 + 250.0
        T_ti = ti.ndarray(dtype=ti.f64, shape=n); T_ti.from_numpy(T)
        res  = ti.ndarray(dtype=ti.f64, shape=n)

        stats_py = get_function_benchmark(lambda: py_buck(T), ops_per_call=n)
        stats_ti = get_function_benchmark(lambda: ti_buck(T), ops_per_call=n)
        stats_k  = get_function_benchmark(
            lambda: kget_buck_vapor_pressure(T_ti, res), ops_per_call=n
        )
        rows.append([n, *stats_py["array_stats"], *stats_ti["array_stats"],
                     *stats_k["array_stats"]])

    python_hdr = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in stats_k["array_headers"]]
    header     = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)
    save_combined_csv(os.path.join(out_dir, "buck_vapor_pressure_benchmark.csv"),
                      header, rows)
    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)
    plot_throughput_vs_array_length(
        header, rows, "Buck vapor-pressure throughput benchmark",
        os.path.join(out_dir, "buck_vapor_pressure_benchmark.png"),
    )

# -------------------------------------------------------------------------
# Entrypoint guard
# -------------------------------------------------------------------------
if __name__ == "__main__":
    benchmark_antoine_vapor_pressure_csv()
    benchmark_clausius_clapeyron_vapor_pressure_csv()
    benchmark_buck_vapor_pressure_csv()
