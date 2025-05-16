"""Benchmarks Python, Taichi wrapper and raw Taichi-kernel versions of
get_aerodynamic_length over a logarithmic range of array sizes."""

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

from particula.particles.properties.aerodynamic_size import (
    get_aerodynamic_length as py_func,
)
from particula.backend.taichi.particles.properties.ti_aerodynamic_length_module import (
    ti_get_aerodynamic_length as ti_func,
    kget_aerodynamic_length as ti_kernel,
)

# --------------------------------------------------------------------------- #
#  CONFIGURATION                                                              #
# --------------------------------------------------------------------------- #
RNG_SEED      = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)   # 10² … 10⁸
ti.init(arch=ti.cpu)


def benchmark_aerodynamic_length_csv() -> None:
    """Run timing benchmark and write CSV, JSON and PNG to ./benchmark_outputs/."""
    rows: list[list[float]] = []
    rng = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        # ----------------------- input data ---------------------------------
        pl  = rng.random(n, dtype=np.float64) * 1e-6 + 1e-9
        psc = rng.random(n, dtype=np.float64) + 1.0
        asc = rng.random(n, dtype=np.float64) + 1.0
        rho = rng.random(n, dtype=np.float64) * 2.0e3 + 1.0e3

        # ----------------------- Taichi buffers -----------------------------
        pl_ti  = ti.ndarray(dtype=ti.f64, shape=n); pl_ti.from_numpy(pl)
        psc_ti = ti.ndarray(dtype=ti.f64, shape=n); psc_ti.from_numpy(psc)
        asc_ti = ti.ndarray(dtype=ti.f64, shape=n); asc_ti.from_numpy(asc)
        rho_ti = ti.ndarray(dtype=ti.f64, shape=n); rho_ti.from_numpy(rho)
        res_ti = ti.ndarray(dtype=ti.f64, shape=n)

        # ----------------------- timing ------------------------------------
        stats_py     = get_function_benchmark(
            lambda: py_func(pl, psc, asc, rho), ops_per_call=n
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_func(pl, psc, asc, rho), ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: ti_kernel(pl_ti, psc_ti, asc_ti, rho_ti,
                              1000.0, 1.0, res_ti), ops_per_call=n
        )

        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    # ----------------------- header & outputs ------------------------------
    python_hdr = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    save_combined_csv(
        os.path.join(out_dir, "aerodynamic_length_benchmark.csv"),
        header,
        rows,
    )

    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        rows,
        "Aerodynamic length throughput benchmark",
        os.path.join(out_dir, "aerodynamic_length_benchmark.png"),
    )


if __name__ == "__main__":
    benchmark_aerodynamic_length_csv()
