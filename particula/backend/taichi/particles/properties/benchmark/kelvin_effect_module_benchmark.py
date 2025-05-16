import os, json, numpy as np, taichi as ti
from particula.backend.benchmark import (
    get_function_benchmark, get_system_info,
    save_combined_csv, plot_throughput_vs_array_length,
)
# Python reference
from particula.particles.properties.kelvin_effect_module import (
    get_kelvin_radius  as py_get_kelvin_radius,
    get_kelvin_term    as py_get_kelvin_term,
)
# Taichi wrapper & kernels
from particula.backend.taichi.particles.properties.ti_kelvin_effect_module import (
    ti_get_kelvin_radius, ti_get_kelvin_term,
    kget_kelvin_radius,  kget_kelvin_term,
)

RNG_SEED      = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)      # 1-D
ARRAY_LENGTHS_2D = np.logspace(2, 4, 4, dtype=int)    # 2-D
ti.init(arch=ti.cpu)

def benchmark_kelvin_radius_csv():
    rows, rng = [], np.random.default_rng(seed=RNG_SEED)
    for n in ARRAY_LENGTHS:
        st = rng.random(n, dtype=np.float64) * 0.02 + 0.05      # σ  (≈0.05-0.07)
        de = rng.random(n, dtype=np.float64) * 200   + 900      # ρ  (≈900-1100)
        mm = rng.random(n, dtype=np.float64) * 0.005 + 0.015    # M  (≈0.015-0.020)
        T  = 298.15

        st_ti = ti.ndarray(dtype=ti.f64, shape=n); st_ti.from_numpy(st)
        de_ti = ti.ndarray(dtype=ti.f64, shape=n); de_ti.from_numpy(de)
        mm_ti = ti.ndarray(dtype=ti.f64, shape=n); mm_ti.from_numpy(mm)
        res_ti= ti.ndarray(dtype=ti.f64, shape=n)

        stats_py  = get_function_benchmark(
            lambda: py_get_kelvin_radius(st, de, mm, T), ops_per_call=n)
        stats_ti  = get_function_benchmark(
            lambda: ti_get_kelvin_radius(st, de, mm, T), ops_per_call=n)
        stats_ker = get_function_benchmark(
            lambda: kget_kelvin_radius(st_ti, de_ti, mm_ti, T, res_ti),
            ops_per_call=n)

        rows.append([n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_ker["array_stats"],
        ])
    _save_outputs("kelvin_radius", rows, stats_py, stats_ti, stats_ker)

def benchmark_kelvin_term_csv():
    rows, rng = [], np.random.default_rng(seed=RNG_SEED)
    for n in ARRAY_LENGTHS_2D:
        pr = rng.random(n, dtype=np.float64) * 5e-8  + 5e-8
        kr = rng.random(n, dtype=np.float64) * 5e-8  + 5e-8
        pr_mat = np.broadcast_to(pr[:, None], (n, n))
        kr_mat = np.broadcast_to(kr[None, :], (n, n))

        pr_ti = ti.ndarray(dtype=ti.f64, shape=pr_mat.shape); pr_ti.from_numpy(pr_mat)
        kr_ti = ti.ndarray(dtype=ti.f64, shape=kr_mat.shape); kr_ti.from_numpy(kr_mat)
        res_ti= ti.ndarray(dtype=ti.f64, shape=pr_mat.shape)

        elems = n * n
        stats_py  = get_function_benchmark(
            lambda: py_get_kelvin_term(pr_mat, kr_mat), ops_per_call=elems)
        stats_ti  = get_function_benchmark(
            lambda: ti_get_kelvin_term(pr_mat, kr_mat), ops_per_call=elems)
        stats_ker = get_function_benchmark(
            lambda: kget_kelvin_term(pr_ti, kr_ti, res_ti),
            ops_per_call=elems)

        rows.append([elems,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_ker["array_stats"],
        ])
    _save_outputs("kelvin_term", rows, stats_py, stats_ti, stats_ker)

def _save_outputs(tag, rows, stats_py, stats_ti, stats_ker):
    py_hdr  = ["python_"        + h for h in stats_py ["array_headers"]]
    ti_hdr  = ["taichi_"        + h for h in stats_ti ["array_headers"]]
    ker_hdr = ["taichi_kernel_" + h for h in stats_ker["array_headers"]]
    header  = ["array_length", *py_hdr, *ti_hdr, *ker_hdr]

    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{tag}_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header, rows, f"{tag} throughput benchmark",
        os.path.join(out_dir, f"{tag}_benchmark.png"),
    )

if __name__ == "__main__":
    benchmark_kelvin_radius_csv()
    benchmark_kelvin_term_csv()
