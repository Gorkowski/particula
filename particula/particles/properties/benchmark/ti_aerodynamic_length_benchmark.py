"""
Benchmarks the reference NumPy implementation, the Taichi wrapper, and the
raw Taichi kernel for get_aerodynamic_length().
"""

# ---------- required imports --------------------------------------------
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
from particula.particles.properties.aerodynamic_size import (   # pure-Python
    get_aerodynamic_length as py_func,
)
from particula.backend.taichi.particles.properties.ti_aerodynamic_length_module import (
    ti_get_aerodynamic_length as ti_func,                      # Taichi wrapper
    kget_aerodynamic_length  as ti_kernel,                     # raw kernel
)

# ---------- fixed configuration -----------------------------------------
RNG_SEED      = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)   # 10² … 10⁸
ti.init(arch=ti.cpu)


def benchmark_ti_aerodynamic_length_csv():
    """Time Python vs. Taichi vs. kernel and store CSV + PNG + JSON."""
    rows = []
    rng  = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        # -------- make random input data --------------------------------
        pl  = rng.random(n, dtype=np.float64) + 1e-9            # physical length
        psc = rng.random(n, dtype=np.float64) + 1e-9            # phys. slip
        asc = rng.random(n, dtype=np.float64) + 1e-9            # aero. slip
        rho = rng.random(n, dtype=np.float64) * 1000 + 500.0    # density > 0
        ref_rho, chi = 1000.0, 1.0                              # defaults

        # -------- Taichi buffers ----------------------------------------
        pl_ti  = ti.ndarray(dtype=ti.f64, shape=n)
        psc_ti = ti.ndarray(dtype=ti.f64, shape=n)
        asc_ti = ti.ndarray(dtype=ti.f64, shape=n)
        rho_ti = ti.ndarray(dtype=ti.f64, shape=n)
        res_ti = ti.ndarray(dtype=ti.f64, shape=n)
        pl_ti.from_numpy(pl); psc_ti.from_numpy(psc)
        asc_ti.from_numpy(asc); rho_ti.from_numpy(rho)

        # -------- timing ------------------------------------------------
        stats_py = get_function_benchmark(
            lambda: py_func(pl, psc, asc, rho), ops_per_call=n
        )
        stats_ti = get_function_benchmark(
            lambda: ti_func(pl, psc, asc, rho), ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: ti_kernel(pl_ti, psc_ti, asc_ti, rho_ti, ref_rho, chi, res_ti),
            ops_per_call=n,
        )

        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    # ---------- CSV / JSON / plot output --------------------------------
    python_hdr = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    save_combined_csv(
        os.path.join(out_dir, "ti_aerodynamic_length_benchmark.csv"),
        header, rows
    )
    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        rows,
        "Aerodynamic Length throughput benchmark",
        os.path.join(out_dir, "ti_aerodynamic_length_benchmark.png"),
    )


if __name__ == "__main__":
    benchmark_ti_aerodynamic_length_csv()
