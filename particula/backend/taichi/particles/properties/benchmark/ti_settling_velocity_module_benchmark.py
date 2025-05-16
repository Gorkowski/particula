"""Benchmarks the reference Python, Taichi wrapper, and raw Taichi kernel."""
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

from particula.particles.properties.settling_velocity import (
    get_particle_settling_velocity as py_func,
)
from particula.backend.taichi.particles.properties.ti_settling_velocity import (
    ti_get_particle_settling_velocity as ti_func,
    kget_particle_settling_velocity  as ti_kernel,
)

# -- fixed RNG and Taichi backend for reproducibility -------------------
RNG_SEED = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)   # 10² … 10⁸
ti.init(arch=ti.cpu)

def benchmark_particle_settling_velocity_csv():
    """
    Time pure-Python, Taichi wrapper, and raw kernel over ARRAY_LENGTHS,
    then save CSV, JSON, and PNG into ./benchmark_outputs/.
    """
    rows = []
    rng  = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        # ---------- random input ---------------------------------------
        r      = rng.random(n, dtype=np.float64) * 2.0e-6 + 0.5e-6    # 0.5–2.5 µm
        rho_p  = rng.random(n, dtype=np.float64) * 1000.0 + 1000.0    # 1 000–2 000 kg/m³
        c_c    = rng.random(n, dtype=np.float64) * 0.5    + 1.0       # 1.0–1.5
        mu     = 1.8e-5                                              # Pa·s
        g      = 9.80665                                             # m/s²
        rho_f  = 1.225                                               # kg/m³

        # ---------- Taichi buffers ------------------------------------
        r_ti     = ti.ndarray(dtype=ti.f64, shape=n); r_ti.from_numpy(r)
        rho_p_ti = ti.ndarray(dtype=ti.f64, shape=n); rho_p_ti.from_numpy(rho_p)
        c_c_ti   = ti.ndarray(dtype=ti.f64, shape=n); c_c_ti.from_numpy(c_c)
        res_ti   = ti.ndarray(dtype=ti.f64, shape=n)

        # ---------- timing -------------------------------------------
        stats_py = get_function_benchmark(
            lambda: py_func(r, rho_p, c_c, mu, g, rho_f), ops_per_call=n
        )
        stats_ti = get_function_benchmark(
            lambda: ti_func(r, rho_p, c_c, mu, g, rho_f), ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: ti_kernel(r_ti, rho_p_ti, c_c_ti, mu, g, rho_f, res_ti),
            ops_per_call=n
        )

        # ---------- collect row --------------------------------------
        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    # ---------- header ----------------------------------------------
    python_hdr = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    # ---------- outputs ---------------------------------------------
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "particle_settling_velocity_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        rows,
        "Particle-settling-velocity throughput benchmark",
        os.path.join(out_dir, "particle_settling_velocity_benchmark.png"),
    )

if __name__ == "__main__":
    benchmark_particle_settling_velocity_csv()
