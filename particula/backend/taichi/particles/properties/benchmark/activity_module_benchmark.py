"""Benchmarks the reference Python, Taichi wrapper, and raw Taichi kernels
of activity_module."""
import os, json
import numpy as np
import taichi as ti

from particula.backend.benchmark import (           # guide §2
    get_function_benchmark, get_system_info,
    save_combined_csv, plot_throughput_vs_array_length,
)

# – reference python impl. –
from particula.particles.properties.activity_module import (
    get_ideal_activity_mass   as py_mass,
    get_ideal_activity_volume as py_vol,
    get_ideal_activity_molar  as py_mol,
    get_kappa_activity        as py_kap,
    get_surface_partial_pressure as py_surf_p,
)

# – taichi wrapper + kernels –
from particula.backend.taichi.particles.properties.ti_activity_module import (
    ti_get_ideal_activity_mass   as ti_mass,
    ti_get_ideal_activity_volume as ti_vol,
    ti_get_ideal_activity_molar  as ti_mol,
    ti_get_kappa_activity        as ti_kap,
    ti_get_surface_partial_pressure as ti_surf_p,

    kget_ideal_activity_mass     as k_mass,
    kget_ideal_activity_volume   as k_vol,
    kget_ideal_activity_molar    as k_mol,
    kget_kappa_activity          as k_kap,
    kget_surface_partial_pressure as k_surf_p,
)

# -- fixed RNG and Taichi backend for reproducibility -------------------
RNG_SEED      = 42
ARRAY_LENGTHS = np.logspace(2, 6, 8, dtype=int)     # 2-D, so a bit smaller
ti.init(arch=ti.cpu)

def _run_benchmark(
    name,
    make_np_args,
    py_func, ti_func, ti_kernel
):
    """Benchmark a single routine (Python, Taichi wrapper, and kernel) over ARRAY_LENGTHS."""
    rows, rng = [], np.random.default_rng(RNG_SEED)
    for n in ARRAY_LENGTHS:
        np_args = make_np_args(n, rng)
        # build *once* the matching Taichi ndarrays
        ti_args, ti_res = [], None
        for a in np_args:
            ti_a = ti.ndarray(dtype=ti.f64, shape=a.shape); ti_a.from_numpy(a)
            ti_args.append(ti_a)
        if ti_kernel is not None:
            ti_res = ti.ndarray(dtype=ti.f64, shape=np_args[0].shape)
            ti_args.append(ti_res)

        stats_py     = get_function_benchmark(lambda: py_func(*np_args),
                                              ops_per_call=np_args[0].size)
        stats_ti     = get_function_benchmark(lambda: ti_func(*np_args),
                                              ops_per_call=np_args[0].size)
        stats_kernel = get_function_benchmark(
            lambda: ti_kernel(*ti_args), ops_per_call=np_args[0].size
        )

        rows.append([n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    python_hdr  = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr  = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr  = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{name}_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header, rows, f"{name} throughput benchmark",
        os.path.join(out_dir, f"{name}_benchmark.png"),
    )

def benchmark_ideal_activity_mass_csv():
    """Benchmark ideal_activity_mass (Python, Taichi wrapper, kernel) and save results."""
    _run_benchmark(
        "ideal_activity_mass",
        lambda n,r: (r.random((1, n), dtype=np.float64)+1e-9, ),
        py_mass, ti_mass, k_mass
    )

def benchmark_ideal_activity_volume_csv():
    """Benchmark ideal_activity_volume (Python, Taichi wrapper, kernel) and save results."""
    _run_benchmark(
        "ideal_activity_volume",
        lambda n,r: (
            r.random((1, n), dtype=np.float64)+1e-9,
            r.random(n,        dtype=np.float64)+1e-9,
        ),
        py_vol, ti_vol, k_vol
    )

def benchmark_ideal_activity_molar_csv():
    """Benchmark ideal_activity_molar (Python, Taichi wrapper, kernel) and save results."""
    _run_benchmark(
        "ideal_activity_molar",
        lambda n,r: (
            r.random((1, n), dtype=np.float64)+1e-9,
            r.random(n,        dtype=np.float64)+1e-9,
        ),
        py_mol, ti_mol, k_mol
    )

def benchmark_kappa_activity_csv():
    """Benchmark kappa_activity (Python, Taichi wrapper, kernel) and save results."""
    _run_benchmark(
        "kappa_activity",
        lambda n,r: (
            r.random((1, n), dtype=np.float64)+1e-9,            # mc
            r.random(n,        dtype=np.float64),               # kappa   (≥0)
            r.random(n,        dtype=np.float64)+1e-9,          # density (>0)
            r.random(n,        dtype=np.float64)+1e-9,          # molar m (>0)
            0,                                                  # water idx
        ),
        py_kap, ti_kap, k_kap
    )

def benchmark_surface_partial_pressure_csv():
    """Benchmark surface_partial_pressure (Python, Taichi wrapper, kernel) and save results."""
    _run_benchmark(
        "surface_partial_pressure",
        lambda n,r: (
            r.random(n, dtype=np.float64)+1e-9,   # pvp  (>0)
            r.random(n, dtype=np.float64),        # act (≥0)
        ),
        py_surf_p, ti_surf_p, k_surf_p
    )

if __name__ == "__main__":
    benchmark_ideal_activity_mass_csv()
    benchmark_ideal_activity_volume_csv()
    benchmark_ideal_activity_molar_csv()
    benchmark_kappa_activity_csv()
    benchmark_surface_partial_pressure_csv()
