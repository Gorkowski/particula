"""Benchmarks python, taichi-wrapper and raw-kernel variants of the four
convert-kappa-volume routines."""
import os, json
import numpy as np
import taichi as ti
from particula.backend.benchmark import (
    get_function_benchmark,
    get_system_info,
    save_combined_csv,
    plot_throughput_vs_array_length,
)
from particula.particles.properties.convert_kappa_volumes import (
    get_solute_volume_from_kappa   as py_get_solute_vol,
    get_water_volume_from_kappa    as py_get_water_vol,
    get_kappa_from_volumes         as py_get_kappa,
    get_water_volume_in_mixture    as py_get_mix_water,
)
from particula.backend.taichi.particles.properties.ti_convert_kappa_volumes import (
    ti_get_solute_volume_from_kappa   as ti_get_solute_vol,
    ti_get_water_volume_from_kappa    as ti_get_water_vol,
    ti_get_kappa_from_volumes         as ti_get_kappa,
    ti_get_water_volume_in_mixture    as ti_get_mix_water,
    kget_solute_volume_from_kappa     as kget_solute_vol,
    kget_water_volume_from_kappa      as kget_water_vol,
    kget_kappa_from_volumes           as kget_kappa,
    kget_water_volume_in_mixture      as kget_mix_water,
)

RNG_SEED      = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)   # 10² … 10⁸
ti.init(arch=ti.cpu)
AW_CONST = np.float64(0.8)     # constant water activity used below

def _vec_py_get_solute_volume_from_kappa(v_tot, kappa, aw):
    vt_arr, kp_arr = np.asarray(v_tot), np.asarray(kappa)
    flat = [
        py_get_solute_vol(v, k, aw) for v, k in zip(vt_arr.ravel(), kp_arr.ravel())
    ]
    return np.reshape(np.array(flat, dtype=np.float64), vt_arr.shape)

def _run_benchmark(rows, n, stats_py, stats_ti, stats_kernel):
    rows.append([
        n,
        *stats_py["array_stats"],
        *stats_ti["array_stats"],
        *stats_kernel["array_stats"],
    ])
    return rows, stats_py, stats_ti, stats_kernel

def benchmark_solute_volume_from_kappa_csv():
    """Time *get_solute_volume_from_kappa* over ARRAY_LENGTHS."""
    rows = []
    rng = np.random.default_rng(RNG_SEED)
    for n in ARRAY_LENGTHS:
        vt = rng.random(n) + 1e-9
        kp = rng.random(n) + 1e-9
        # Taichi buffers
        vt_t = ti.ndarray(dtype=ti.f64, shape=n); vt_t.from_numpy(vt)
        kp_t = ti.ndarray(dtype=ti.f64, shape=n); kp_t.from_numpy(kp)
        res_t = ti.ndarray(dtype=ti.f64, shape=n)
        # timings
        stats_py     = get_function_benchmark(
            lambda: _vec_py_get_solute_volume_from_kappa(vt, kp, AW_CONST),
            ops_per_call=n,
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_get_solute_vol(vt, kp, AW_CONST),
            ops_per_call=n,
        )
        stats_kernel = get_function_benchmark(
            lambda: kget_solute_vol(vt_t, kp_t, AW_CONST, res_t),
            ops_per_call=n,
        )
        rows, *_ = _run_benchmark(rows, n, stats_py, stats_ti, stats_kernel)

    python_hdr  = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr  = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr  = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "solute_volume_from_kappa_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        rows,
        "Solute volume from κ benchmark",
        os.path.join(out_dir, "solute_volume_from_kappa_benchmark.png"),
    )

def benchmark_water_volume_from_kappa_csv():
    """Time *get_water_volume_from_kappa* over ARRAY_LENGTHS."""
    rows = []
    rng = np.random.default_rng(RNG_SEED)
    for n in ARRAY_LENGTHS:
        vs = rng.random(n) + 1e-9
        kp = rng.random(n) + 1e-9
        vs_t = ti.ndarray(dtype=ti.f64, shape=n); vs_t.from_numpy(vs)
        kp_t = ti.ndarray(dtype=ti.f64, shape=n); kp_t.from_numpy(kp)
        res_t = ti.ndarray(dtype=ti.f64, shape=n)
        stats_py     = get_function_benchmark(
            lambda: py_get_water_vol(vs, kp, AW_CONST),
            ops_per_call=n,
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_get_water_vol(vs, kp, AW_CONST),
            ops_per_call=n,
        )
        stats_kernel = get_function_benchmark(
            lambda: kget_water_vol(vs_t, kp_t, AW_CONST, res_t),
            ops_per_call=n,
        )
        rows, *_ = _run_benchmark(rows, n, stats_py, stats_ti, stats_kernel)

    python_hdr  = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr  = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr  = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "water_volume_from_kappa_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        rows,
        "Water volume from κ benchmark",
        os.path.join(out_dir, "water_volume_from_kappa_benchmark.png"),
    )

def benchmark_kappa_from_volumes_csv():
    """Time *get_kappa_from_volumes* over ARRAY_LENGTHS."""
    rows = []
    rng = np.random.default_rng(RNG_SEED)
    for n in ARRAY_LENGTHS:
        vs = rng.random(n) + 1e-9
        vw = rng.random(n) + 1e-9
        vs_t = ti.ndarray(dtype=ti.f64, shape=n); vs_t.from_numpy(vs)
        vw_t = ti.ndarray(dtype=ti.f64, shape=n); vw_t.from_numpy(vw)
        res_t = ti.ndarray(dtype=ti.f64, shape=n)
        stats_py     = get_function_benchmark(
            lambda: py_get_kappa(vs, vw, AW_CONST),
            ops_per_call=n,
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_get_kappa(vs, vw, AW_CONST),
            ops_per_call=n,
        )
        stats_kernel = get_function_benchmark(
            lambda: kget_kappa(vs_t, vw_t, AW_CONST, res_t),
            ops_per_call=n,
        )
        rows, *_ = _run_benchmark(rows, n, stats_py, stats_ti, stats_kernel)

    python_hdr  = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr  = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr  = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "kappa_from_volumes_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        rows,
        "Kappa from volumes benchmark",
        os.path.join(out_dir, "kappa_from_volumes_benchmark.png"),
    )

def benchmark_water_volume_in_mixture_csv():
    """Time *get_water_volume_in_mixture* over ARRAY_LENGTHS."""
    rows = []
    rng = np.random.default_rng(RNG_SEED)
    for n in ARRAY_LENGTHS:
        vsd = rng.random(n) + 1e-9
        phi = rng.random(n)
        vsd_t = ti.ndarray(dtype=ti.f64, shape=n); vsd_t.from_numpy(vsd)
        phi_t = ti.ndarray(dtype=ti.f64, shape=n); phi_t.from_numpy(phi)
        res_t = ti.ndarray(dtype=ti.f64, shape=n)
        stats_py     = get_function_benchmark(
            lambda: py_get_mix_water(vsd, phi),
            ops_per_call=n,
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_get_mix_water(vsd, phi),
            ops_per_call=n,
        )
        stats_kernel = get_function_benchmark(
            lambda: kget_mix_water(vsd_t, phi_t, res_t),
            ops_per_call=n,
        )
        rows, *_ = _run_benchmark(rows, n, stats_py, stats_ti, stats_kernel)

    python_hdr  = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr  = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr  = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "water_volume_in_mixture_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        rows,
        "Water volume in mixture benchmark",
        os.path.join(out_dir, "water_volume_in_mixture_benchmark.png"),
    )

if __name__ == "__main__":
    benchmark_solute_volume_from_kappa_csv()
    benchmark_water_volume_from_kappa_csv()
    benchmark_kappa_from_volumes_csv()
    benchmark_water_volume_in_mixture_csv()
