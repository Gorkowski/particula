"""
Benchmarks the reference Python, Taichi-wrapper and Taichi-kernel
implementations of the three mass-concentration-conversion routines.
"""
import os, json, numpy as np, taichi as ti
from particula.backend.benchmark import (
    get_function_benchmark, get_system_info,
    save_combined_csv, plot_throughput_vs_array_length,
)

# ­­­ reference Python functions ­­­
from particula.particles.properties.convert_mass_concentration import (
    get_mole_fraction_from_mass   as py_mole,
    get_volume_fraction_from_mass as py_vol,
    get_mass_fraction_from_mass   as py_mass,
)

# ­­­ Taichi wrappers & kernels ­­­
from particula.backend.taichi.particles.properties.ti_convert_mass_concentration_module import (
    ti_get_mole_fraction_from_mass   as ti_mole,
    kget_mole_fraction_from_mass     as k_mole,
    ti_get_volume_fraction_from_mass as ti_vol,
    kget_volume_fraction_from_mass   as k_vol,
    ti_get_mass_fraction_from_mass   as ti_mass,
    kget_mass_fraction_from_mass     as k_mass,
)

RNG_SEED      = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)      # 10² … 10⁸
ti.init(arch=ti.cpu)

def benchmark_mole_fraction_csv() -> None:
    """Time get_mole_fraction_from_mass across ARRAY_LENGTHS and save CSV/PNG."""
    rows, rng = [], np.random.default_rng(RNG_SEED)
    for n in ARRAY_LENGTHS:
        m  = rng.random(n, dtype=np.float64) + 1e-12   # kg m⁻³
        mm = rng.random(n, dtype=np.float64) + 1e-12   # kg mol⁻¹

        m_ti  = ti.ndarray(dtype=ti.f64, shape=n);  m_ti.from_numpy(m)
        mm_ti = ti.ndarray(dtype=ti.f64, shape=n); mm_ti.from_numpy(mm)
        out   = ti.ndarray(dtype=ti.f64, shape=n)

        stats_py     = get_function_benchmark(lambda:  py_mole(m,  mm), ops_per_call=n)
        stats_ti     = get_function_benchmark(lambda:  ti_mole(m, mm), ops_per_call=n)
        stats_kernel = get_function_benchmark(lambda: k_mole(m_ti, mm_ti, out), ops_per_call=n)

        rows.append([n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    python_hdr = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)
    csv_file = os.path.join(out_dir, "mole_fraction_from_mass_benchmark.csv")
    save_combined_csv(csv_file, header, rows)

    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header, rows, "Mole-fraction-from-mass throughput benchmark",
        os.path.join(out_dir, "mole_fraction_from_mass_benchmark.png"),
    )

def benchmark_volume_fraction_csv() -> None:
    """Time get_volume_fraction_from_mass across ARRAY_LENGTHS and save CSV/PNG."""
    rows, rng = [], np.random.default_rng(RNG_SEED)
    for n in ARRAY_LENGTHS:
        m   = rng.random(n, dtype=np.float64) + 1e-12   # kg m⁻³
        rho = rng.random(n, dtype=np.float64) + 1e-12   # kg m⁻³

        m_ti   = ti.ndarray(dtype=ti.f64, shape=n);   m_ti.from_numpy(m)
        rho_ti = ti.ndarray(dtype=ti.f64, shape=n); rho_ti.from_numpy(rho)
        out    = ti.ndarray(dtype=ti.f64, shape=n)

        stats_py     = get_function_benchmark(lambda:  py_vol(m,  rho), ops_per_call=n)
        stats_ti     = get_function_benchmark(lambda:  ti_vol(m, rho), ops_per_call=n)
        stats_kernel = get_function_benchmark(lambda: k_vol(m_ti, rho_ti, out), ops_per_call=n)

        rows.append([n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    python_hdr = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)
    csv_file = os.path.join(out_dir, "volume_fraction_from_mass_benchmark.csv")
    save_combined_csv(csv_file, header, rows)

    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header, rows, "Volume-fraction-from-mass throughput benchmark",
        os.path.join(out_dir, "volume_fraction_from_mass_benchmark.png"),
    )

def benchmark_mass_fraction_csv() -> None:
    """Time get_mass_fraction_from_mass across ARRAY_LENGTHS and save CSV/PNG."""
    rows, rng = [], np.random.default_rng(RNG_SEED)
    for n in ARRAY_LENGTHS:
        m = rng.random(n, dtype=np.float64) + 1e-12   # kg m⁻³

        m_ti = ti.ndarray(dtype=ti.f64, shape=n); m_ti.from_numpy(m)
        out  = ti.ndarray(dtype=ti.f64, shape=n)

        stats_py     = get_function_benchmark(lambda:  py_mass(m), ops_per_call=n)
        stats_ti     = get_function_benchmark(lambda:  ti_mass(m), ops_per_call=n)
        stats_kernel = get_function_benchmark(lambda: k_mass(m_ti, out), ops_per_call=n)

        rows.append([n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    python_hdr = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)
    csv_file = os.path.join(out_dir, "mass_fraction_from_mass_benchmark.csv")
    save_combined_csv(csv_file, header, rows)

    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header, rows, "Mass-fraction-from-mass throughput benchmark",
        os.path.join(out_dir, "mass_fraction_from_mass_benchmark.png"),
    )

if __name__ == "__main__":
    benchmark_mole_fraction_csv()
    benchmark_volume_fraction_csv()
    benchmark_mass_fraction_csv()
