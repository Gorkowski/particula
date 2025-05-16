"""Benchmarks pure-Python, Taichi wrapper, and raw Taichi kernel."""
import os, json, numpy as np, taichi as ti
from particula.backend.benchmark import (
    get_function_benchmark, get_system_info,
    save_combined_csv, plot_throughput_vs_array_length,
)
# reference, wrapper & kernel
from particula.particles.properties.inertia_time import (
    get_particle_inertia_time as py_func,
)
from particula.backend.taichi.particles.properties.ti_inertia_time_module import (
    ti_get_particle_inertia_time as ti_func,
    kget_particle_inertia_time  as ti_kernel,
)
RNG_SEED = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)
ti.init(arch=ti.cpu)          # CPU only – reproducible

def benchmark_ti_inertia_time_csv():
    """Time all three implementations over ARRAY_LENGTHS and save outputs."""
    rows, rng = [], np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        # -------- random positive test data ----------------------------
        r     = rng.random(n) * 1e-5 + 1e-9        # m
        rho_p = rng.random(n) * 3000.0 + 500.0     # kg/m³
        rho_f = rng.random(n) *  5.0  +   1.0      # kg/m³
        nu    = rng.random(n) * 1e-4  + 1e-6       # m²/s

        # -------- Taichi buffers ---------------------------------------
        r_ti, rho_p_ti, rho_f_ti, nu_ti, res_ti = [
            ti.ndarray(dtype=ti.f64, shape=n) for _ in range(5)
        ]
        for np_arr, ti_arr in ((r, r_ti), (rho_p, rho_p_ti),
                               (rho_f, rho_f_ti), (nu, nu_ti)):
            ti_arr.from_numpy(np_arr)

        # -------- timing -----------------------------------------------
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
        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    # -------- headers & output paths -----------------------------------
    python_hdr  = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr  = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr  = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "ti_inertia_time_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        rows,
        "Particle inertia time throughput benchmark",
        os.path.join(out_dir, "ti_inertia_time_benchmark.png"),
    )

if __name__ == "__main__":
    benchmark_ti_inertia_time_csv()
