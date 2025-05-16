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
from particula.gas.properties.dynamic_viscosity import (
    get_dynamic_viscosity as py_func,
)
from particula.backend.taichi.gas.properties.ti_dynamic_viscosity_module import (
    ti_get_dynamic_viscosity as ti_func,
    kget_dynamic_viscosity   as ti_kernel,
)
from particula.util.constants import (
    REF_TEMPERATURE_STP,
    REF_VISCOSITY_AIR_STP,
)

# -- fixed RNG and Taichi backend for reproducibility -------------------
RNG_SEED = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)   # 10² … 10⁸
ti.init(arch=ti.cpu)

def benchmark_dynamic_viscosity_csv():
    """
    Time pure-Python, Taichi wrapper, and raw kernel over ARRAY_LENGTHS,
    then save CSV, JSON, and PNG into ./benchmark_outputs/.
    """
    rows = []
    rng  = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        # ----- random input data -----------------------------------------
        temps = rng.random(n, dtype=np.float64) * 150.0 + 200.0   # 200-350 K
        rv    = np.full(n, REF_VISCOSITY_AIR_STP, dtype=np.float64)
        rt    = np.full(n, REF_TEMPERATURE_STP, dtype=np.float64)

        # ----- Taichi buffers --------------------------------------------
        temps_ti = ti.ndarray(dtype=ti.f64, shape=n)
        rv_ti    = ti.ndarray(dtype=ti.f64, shape=n)
        rt_ti    = ti.ndarray(dtype=ti.f64, shape=n)
        res_ti   = ti.ndarray(dtype=ti.f64, shape=n)
        temps_ti.from_numpy(temps)
        rv_ti.from_numpy(rv)
        rt_ti.from_numpy(rt)

        # ----- timing ----------------------------------------------------
        stats_py     = get_function_benchmark(
            lambda: py_func(temps), ops_per_call=n)
        stats_ti     = get_function_benchmark(
            lambda: ti_func(temps), ops_per_call=n)
        stats_kernel = get_function_benchmark(
            lambda: ti_kernel(temps_ti, rv_ti, rt_ti, res_ti),
            ops_per_call=n)

        # ----- collect row -----------------------------------------------
        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    # ---------------- header --------------------------------------------
    python_hdr  = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr  = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr  = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    # ---------------- output dir ----------------------------------------
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    # ---------------- CSV -----------------------------------------------
    save_combined_csv(
        os.path.join(out_dir, "dynamic_viscosity_benchmark.csv"),
        header, rows)

    # ---------------- system info JSON ----------------------------------
    with open(os.path.join(out_dir, "system_info.json"), "w",
              encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    # ---------------- throughput plot -----------------------------------
    plot_throughput_vs_array_length(
        header, rows,
        "Dynamic viscosity throughput benchmark",
        os.path.join(out_dir, "dynamic_viscosity_benchmark.png"),
    )

if __name__ == "__main__":
    benchmark_dynamic_viscosity_csv()
