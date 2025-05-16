"""Benchmarks the collision-radius routines
(pure Python vs. Taichi wrapper vs. raw Taichi kernel)."""

# --------------------------- required imports ---------------------------
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

from particula.particles.properties.collision_radius_module import (
    get_collision_radius_mg1988 as py_get_mg1988,
    get_collision_radius_sr1992 as py_get_sr1992,
)
from particula.backend.taichi.particles.properties.ti_collision_radius_module import (
    ti_get_collision_radius_mg1988 as ti_get_mg1988,
    ti_get_collision_radius_sr1992 as ti_get_sr1992,
    kget_collision_radius_mg1988 as kget_mg1988,
    kget_collision_radius_sr1992 as kget_sr1992,
)

# --------------------------- benchmark config ---------------------------
RNG_SEED = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)        # 10² … 10⁸
ti.init(arch=ti.cpu)


def _out_dir() -> str:
    """Return ./benchmark_outputs directory and create it if necessary."""
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# --------------------------- MG1988 routine -----------------------------
def benchmark_collision_radius_mg1988_csv():
    """
    Time Python, Taichi wrapper, and raw Taichi kernel for the MG1988
    collision-radius routine and write CSV/JSON/PNG to ./benchmark_outputs/.
    """
    rows, rng = [], np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        r_g = rng.random(n, dtype=np.float64) + 1e-9      # avoid zero
        r_g_ti = ti.ndarray(dtype=ti.f64, shape=n)
        out_ti = ti.ndarray(dtype=ti.f64, shape=n)
        r_g_ti.from_numpy(r_g)

        stats_py  = get_function_benchmark(lambda: py_get_mg1988(r_g),          ops_per_call=n)
        stats_ti  = get_function_benchmark(lambda: ti_get_mg1988(r_g),          ops_per_call=n)
        stats_ke  = get_function_benchmark(lambda: kget_mg1988(r_g_ti, out_ti), ops_per_call=n)

        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_ke["array_stats"],
        ])

    header = ["array_length",
              *["python_"        + h for h in stats_py["array_headers"]],
              *["taichi_"        + h for h in stats_ti["array_headers"]],
              *["taichi_kernel_" + h for h in stats_ke["array_headers"]]]

    out_dir = _out_dir()
    save_combined_csv(
        os.path.join(out_dir, "collision_radius_mg1988_benchmark.csv"),
        header,
        rows,
    )
    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        rows,
        "Collision-radius MG1988 throughput benchmark",
        os.path.join(out_dir, "collision_radius_mg1988_benchmark.png"),
    )


# --------------------------- SR1992 routine -----------------------------
def benchmark_collision_radius_sr1992_csv():
    """
    Time Python, Taichi wrapper, and raw Taichi kernel for the SR1992
    collision-radius routine and write CSV/JSON/PNG to ./benchmark_outputs/.
    """
    rows, rng = [], np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        r_g = rng.random(n, dtype=np.float64) + 1.0
        d_f = rng.random(n, dtype=np.float64) + 1.0
        r_g_ti = ti.ndarray(dtype=ti.f64, shape=n);  r_g_ti.from_numpy(r_g)
        d_f_ti = ti.ndarray(dtype=ti.f64, shape=n);  d_f_ti.from_numpy(d_f)
        out_ti = ti.ndarray(dtype=ti.f64, shape=n)

        stats_py = get_function_benchmark(lambda: py_get_sr1992(r_g, d_f),                 ops_per_call=n)
        stats_ti = get_function_benchmark(lambda: ti_get_sr1992(r_g, d_f),                 ops_per_call=n)
        stats_ke = get_function_benchmark(lambda: kget_sr1992(r_g_ti, d_f_ti, out_ti),     ops_per_call=n)

        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_ke["array_stats"],
        ])

    header = ["array_length",
              *["python_"        + h for h in stats_py["array_headers"]],
              *["taichi_"        + h for h in stats_ti["array_headers"]],
              *["taichi_kernel_" + h for h in stats_ke["array_headers"]]]

    out_dir = _out_dir()
    save_combined_csv(
        os.path.join(out_dir, "collision_radius_sr1992_benchmark.csv"),
        header,
        rows,
    )
    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        rows,
        "Collision-radius SR1992 throughput benchmark",
        os.path.join(out_dir, "collision_radius_sr1992_benchmark.png"),
    )


# --------------------------- entry-point guard --------------------------
if __name__ == "__main__":
    benchmark_collision_radius_mg1988_csv()
    benchmark_collision_radius_sr1992_csv()
