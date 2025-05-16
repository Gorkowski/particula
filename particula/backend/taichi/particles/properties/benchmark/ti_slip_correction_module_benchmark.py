"""Benchmarks Python vs. Taichi Cunningham slip-correction implementations."""
# –– required imports ––––––––––––––––––––––––––––––––––––––––––––––––––––––
import os, json
import numpy as np
import taichi as ti
from particula.backend.benchmark import (
    get_function_benchmark,
    get_system_info,
    save_combined_csv,
    plot_throughput_vs_array_length,
)
from particula.particles.properties.slip_correction_module import (
    get_cunningham_slip_correction as py_func,
)
from particula.backend.taichi.particles.properties.ti_slip_correction_module import (
    ti_get_cunningham_slip_correction as ti_func,
    kget_cunningham_slip_correction   as ti_kernel,
)
# –– reproducibility / backend –––––––––––––––––––––––––––––––––––––––––––––
RNG_SEED      = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)      # 10² … 10⁸
ti.init(arch=ti.cpu)
# –– main benchmark ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
def benchmark_cunningham_slip_correction_csv() -> None:
    """
    Time pure-Python, Taichi wrapper, and raw kernel over ARRAY_LENGTHS,
    then save CSV, JSON, and PNG into ./benchmark_outputs/.
    """
    rows: list[list[float]] = []
    rng  = np.random.default_rng(seed=RNG_SEED)
    for n in ARRAY_LENGTHS:
        # –– random input data ––––––––––––––––––––––––––––––––––––––––––––
        kn = rng.random(n, dtype=np.float64) * 1.95 + 0.05   # 0.05 … 2.0
        # –– Taichi buffers ––––––––––––––––––––––––––––––––––––––––––––––
        kn_ti  = ti.ndarray(dtype=ti.f64, shape=n)
        res_ti = ti.ndarray(dtype=ti.f64, shape=n)
        kn_ti.from_numpy(kn)
        # –– timing ––––––––––––––––––––––––––––––––––––––––––––––––––––––
        stats_py     = get_function_benchmark(lambda: py_func(kn),        ops_per_call=n)
        stats_ti     = get_function_benchmark(lambda: ti_func(kn),        ops_per_call=n)
        stats_kernel = get_function_benchmark(lambda: ti_kernel(kn_ti, res_ti), ops_per_call=n)
        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])
    # –– header ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    python_hdr  = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr  = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr  = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]
    # –– output paths ––––––––––––––––––––––––––––––––––––––––––––––––––––
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "ti_slip_correction_module_benchmark.csv")
    save_combined_csv(csv_path, header, rows)
    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)
    plot_throughput_vs_array_length(
        header,
        rows,
        "Cunningham slip-correction throughput benchmark",
        os.path.join(out_dir, "ti_slip_correction_module_benchmark.png"),
    )
# –– entry-point guard ––––––––––––––––––––––––––––––––––––––––––––––––––––
if __name__ == "__main__":
    benchmark_cunningham_slip_correction_csv()
