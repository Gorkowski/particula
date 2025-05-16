import os, json
import numpy as np
import taichi as ti
from particula.backend.benchmark import (
    get_function_benchmark,
    get_system_info,
    save_combined_csv,
    plot_throughput_vs_array_length,
)
# reference-python functions
from particula.particles.properties.activity_module import (
    get_surface_partial_pressure      as py_spp,
    get_ideal_activity_mass           as py_act_mass,
    get_ideal_activity_volume         as py_act_vol,
    get_ideal_activity_molar          as py_act_mol,
    get_kappa_activity                as py_act_kap,
)
# Taichi wrapper + raw-kernel functions
from particula.backend.taichi.particles.properties.ti_activity_module import (
    ti_get_surface_partial_pressure   as ti_spp,
    ti_get_ideal_activity_mass        as ti_act_mass,
    ti_get_ideal_activity_volume      as ti_act_vol,
    ti_get_ideal_activity_molar       as ti_act_mol,
    ti_get_kappa_activity             as ti_act_kap,
    kget_surface_partial_pressure     as k_spp,
    kget_ideal_activity_mass          as k_act_mass,
    kget_ideal_activity_volume        as k_act_vol,
    kget_ideal_activity_molar         as k_act_mol,
    kget_kappa_activity               as k_act_kap,
)

RNG_SEED = 42
ARRAY_LENGTHS_1D = np.logspace(2, 8, 10, dtype=int)    # for 1-D inputs
ARRAY_LENGTHS_2D = np.logspace(2, 7,  6, dtype=int)    # for 2-D inputs
N_SPECIES = 3                                           # fixed column count
ti.init(arch=ti.cpu)

def _write_results(name:str, header, rows):
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{name}_benchmark.csv")
    save_combined_csv(csv_path, header, rows)
    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)
    plot_throughput_vs_array_length(header, rows,
        f"{name.replace('_',' ').title()} throughput benchmark",
        os.path.join(out_dir, f"{name}_benchmark.png"))

def benchmark_surface_partial_pressure_csv():
    """Benchmark get_surface_partial_pressure (py / ti / kernel)."""
    rows = []
    rng  = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS_1D:
        pvp = rng.random(n, dtype=np.float64) + 1e-9
        act = rng.random(n, dtype=np.float64)

        pvp_ti = ti.ndarray(dtype=ti.f64, shape=n)
        act_ti = ti.ndarray(dtype=ti.f64, shape=n)
        res_ti = ti.ndarray(dtype=ti.f64, shape=n)
        pvp_ti.from_numpy(pvp)
        act_ti.from_numpy(act)

        stats_py     = get_function_benchmark(lambda: py_spp(pvp, act),           ops_per_call=n)
        stats_ti     = get_function_benchmark(lambda: ti_spp(pvp, act),           ops_per_call=n)
        stats_kernel = get_function_benchmark(lambda: k_spp(pvp_ti, act_ti, res_ti),
                                              ops_per_call=n)

        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    python_hdr = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header     = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    _write_results("surface_partial_pressure", header, rows)

def benchmark_ideal_activity_mass_csv():
    """Benchmark get_ideal_activity_mass (py / ti / kernel)."""
    rows = []
    rng  = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS_2D:
        mc = rng.random((n, N_SPECIES), dtype=np.float64) + 1e-9

        mc_ti = ti.ndarray(dtype=ti.f64, shape=(n, N_SPECIES))
        res_ti = ti.ndarray(dtype=ti.f64, shape=(n, N_SPECIES))
        mc_ti.from_numpy(mc)

        stats_py     = get_function_benchmark(lambda: py_act_mass(mc),           ops_per_call=n*N_SPECIES)
        stats_ti     = get_function_benchmark(lambda: ti_act_mass(mc),           ops_per_call=n*N_SPECIES)
        stats_kernel = get_function_benchmark(lambda: k_act_mass(mc_ti, res_ti), ops_per_call=n*N_SPECIES)

        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    python_hdr = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header     = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    _write_results("ideal_activity_mass", header, rows)

def benchmark_ideal_activity_volume_csv():
    """Benchmark get_ideal_activity_volume (py / ti / kernel)."""
    rows = []
    rng  = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS_2D:
        mc = rng.random((n, N_SPECIES), dtype=np.float64) + 1e-9
        dens = rng.random(N_SPECIES, dtype=np.float64) + 1e3

        mc_ti = ti.ndarray(dtype=ti.f64, shape=(n, N_SPECIES))
        dens_ti = ti.ndarray(dtype=ti.f64, shape=N_SPECIES)
        res_ti = ti.ndarray(dtype=ti.f64, shape=(n, N_SPECIES))
        mc_ti.from_numpy(mc)
        dens_ti.from_numpy(dens)

        stats_py     = get_function_benchmark(lambda: py_act_vol(mc, dens),           ops_per_call=n*N_SPECIES)
        stats_ti     = get_function_benchmark(lambda: ti_act_vol(mc, dens),           ops_per_call=n*N_SPECIES)
        stats_kernel = get_function_benchmark(lambda: k_act_vol(mc_ti, dens_ti, res_ti), ops_per_call=n*N_SPECIES)

        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    python_hdr = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header     = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    _write_results("ideal_activity_volume", header, rows)

def benchmark_ideal_activity_molar_csv():
    """Benchmark get_ideal_activity_molar (py / ti / kernel)."""
    rows = []
    rng  = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS_2D:
        mc = rng.random((n, N_SPECIES), dtype=np.float64) + 1e-9
        mm = rng.random(N_SPECIES, dtype=np.float64) + 10.0

        mc_ti = ti.ndarray(dtype=ti.f64, shape=(n, N_SPECIES))
        mm_ti = ti.ndarray(dtype=ti.f64, shape=N_SPECIES)
        res_ti = ti.ndarray(dtype=ti.f64, shape=(n, N_SPECIES))
        mc_ti.from_numpy(mc)
        mm_ti.from_numpy(mm)

        stats_py     = get_function_benchmark(lambda: py_act_mol(mc, mm),           ops_per_call=n*N_SPECIES)
        stats_ti     = get_function_benchmark(lambda: ti_act_mol(mc, mm),           ops_per_call=n*N_SPECIES)
        stats_kernel = get_function_benchmark(lambda: k_act_mol(mc_ti, mm_ti, res_ti), ops_per_call=n*N_SPECIES)

        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    python_hdr = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header     = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    _write_results("ideal_activity_molar", header, rows)

def benchmark_kappa_activity_csv():
    """Benchmark get_kappa_activity (py / ti / kernel)."""
    rows = []
    rng  = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS_2D:
        mc = rng.random((n, N_SPECIES), dtype=np.float64) + 1e-9
        kap = rng.random(N_SPECIES, dtype=np.float64)
        dens = rng.random(N_SPECIES, dtype=np.float64) + 1e3
        mm = rng.random(N_SPECIES, dtype=np.float64) + 10.0
        water_index = 0

        mc_ti = ti.ndarray(dtype=ti.f64, shape=(n, N_SPECIES))
        kap_ti = ti.ndarray(dtype=ti.f64, shape=N_SPECIES)
        dens_ti = ti.ndarray(dtype=ti.f64, shape=N_SPECIES)
        mm_ti = ti.ndarray(dtype=ti.f64, shape=N_SPECIES)
        res_ti = ti.ndarray(dtype=ti.f64, shape=(n, N_SPECIES))
        mc_ti.from_numpy(mc)
        kap_ti.from_numpy(kap)
        dens_ti.from_numpy(dens)
        mm_ti.from_numpy(mm)

        stats_py     = get_function_benchmark(lambda: py_act_kap(mc, kap, dens, mm, water_index),           ops_per_call=n*N_SPECIES)
        stats_ti     = get_function_benchmark(lambda: ti_act_kap(mc, kap, dens, mm, water_index),           ops_per_call=n*N_SPECIES)
        stats_kernel = get_function_benchmark(lambda: k_act_kap(mc_ti, kap_ti, dens_ti, mm_ti, water_index, res_ti), ops_per_call=n*N_SPECIES)

        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    python_hdr = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header     = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    _write_results("kappa_activity", header, rows)

if __name__ == "__main__":
    benchmark_surface_partial_pressure_csv()
    benchmark_ideal_activity_mass_csv()
    benchmark_ideal_activity_volume_csv()
    benchmark_ideal_activity_molar_csv()
    benchmark_kappa_activity_csv()
