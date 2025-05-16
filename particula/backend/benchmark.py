"""
Benchmarking module for timing and comparing the performance of different
accelerated implementations.
"""

import time
import statistics
import gc
from typing import Callable, Optional, Any, Dict
import csv
import platform

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import particula as par


def plot_throughput_vs_array_length(
    header: list[str],
    rows: list[list],
    title: str,
    output_path: str,
    array_length_key: str = "array_length",
) -> None:
    """
    Plot array-length vs *_throughput_calls_per_s for every column that
    matches the suffix and save the figure as *output_path* (PNG).
    """
    # indices ---------------------------------------------------------------
    try:
        idx_len = header.index(array_length_key)
    except ValueError as exc:
        raise ValueError(f"'{array_length_key}' not found in header") from exc

    thrpt_indices = [
        i for i, h in enumerate(header)
        if h.endswith("throughput_calls_per_s") and i != idx_len
    ]
    if not thrpt_indices:
        raise ValueError("No throughput columns found in header")

    # data ------------------------------------------------------------------
    array_lengths = [row[idx_len] for row in rows]

    plt.figure()
    markers = "os^v><dph"  # cycle through a few markers
    for k, idx in enumerate(thrpt_indices):
        values = [row[idx] for row in rows]
        label = header[idx].removesuffix("_throughput_calls_per_s")
        plt.loglog(array_lengths, values, markers[k % len(markers)] + "-", label=label)

    # cosmetics -------------------------------------------------------------
    plt.xlabel("Array length")
    plt.ylabel("Throughput (calls/s)")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# non-standard libraries for this script
import psutil


def get_system_info():
    """
    Gather basic CPU, OS and Python-runtime information.

    Arguments:
        - None.

    Returns:
        - Dictionary mapping descriptive keys to collected values.

    Examples:
        ```py
        info = collect_system_info()
        print(info["total_cores"])
        ```
    """
    info = {}

    # CPU counts
    info["physical_cores"] = psutil.cpu_count(logical=False)
    info["total_cores"] = psutil.cpu_count(logical=True)

    # CPU frequencies
    freq = psutil.cpu_freq()
    info["max_frequency_mhz"] = freq.max
    info["min_frequency_mhz"] = freq.min
    info["current_frequency_mhz"] = freq.current

    # CPU usage
    info["cpu_usage_per_core_%"] = psutil.cpu_percent(percpu=True, interval=1)
    info["total_cpu_usage_%"] = psutil.cpu_percent()

    # OS / machine info
    uname = platform.uname()
    info["system"] = uname.system
    info["release"] = uname.release
    info["version"] = uname.version
    info["machine"] = uname.machine
    info["processor"] = uname.processor

    # Python runtime
    info["python_version"] = platform.python_version()
    info["python_build"] = platform.python_build()
    info["python_compiler"] = platform.python_compiler()

    return info


# pylint: disable=too-many-locals, too-many-statements
def get_function_benchmark(
    func: Callable[[], Any],
    ops_per_call: float,
    max_run_time_s: float = 1.0,
    min_iterations: int = 5,
    repeats: Optional[int] = None,
) -> dict[str, Any]:
    """
    Benchmark a zero-argument function using perf_counter_ns, with GC disabled.

    This function times repeated calls to a no-argument callable, collecting
    statistics on execution time and estimating throughput and efficiency.
    It adapts the number of repeats to ensure reliable timing, and returns
    a dictionary of timing and performance metrics.

    Arguments:
        - func : A no-argument callable (e.g. `lambda: work(x, y)`).
        - ops_per_call : Estimated floating-point operations per call.
        - max_run_time_s : If `repeats` is None, run until this many seconds
          elapse.
        - min_iterations : If `repeats` is None, do at least this many calls.
        - repeats : If set, run exactly this many calls.

    Returns:
        - Dictionary with timing and performance statistics, including:
            - min_time_s : Minimum time per call (seconds).
            - max_time_s : Maximum time per call (seconds).
            - mean_time_s : Mean time per call (seconds).
            - mode_time_s : Mode of time per call (seconds).
            - median_time_s : Median time per call (seconds).
            - std_time_s : Standard deviation of time per call (seconds).
            - throughput_calls_per_s : Calls per second (1 / min_time_s).
            - cycles_per_call : CPU cycles per call (min_time_s × CPU_Hz).
            - flops_per_call : Floating-point operations per call.
            - flops_per_cycle : FLOPs per CPU cycle.
            - function_calls : Number of function calls performed.
            - report : Human-readable summary string.
            - array_stats : List of all statistics above.
            - array_headers : List of corresponding header strings.

            Value types are mixed (float, int, str, list, etc).

    Examples:
        ```py title="Benchmark a trivial lambda"
        stats = benchmark_timer(lambda: sum([1, 2, 3]), ops_per_call=3)
        print(stats["report"])
        ```

    References:
        - "time.perf_counter_ns — Python documentation",
          https://docs.python.org/3/library/time.html#time.perf_counter_ns
    """
    # disable GC for cleaner timing
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()

    timings: list[float] = []
    try:
        if repeats is None:
            start_global = time.perf_counter_ns()
            while (len(timings) < min_iterations) or (
                (time.perf_counter_ns() - start_global) / 1e9 < max_run_time_s
            ):
                t0 = time.perf_counter_ns()
                func()
                t1 = time.perf_counter_ns()
                timings.append((t1 - t0) / 1e9)
        else:
            for _ in range(repeats):
                t0 = time.perf_counter_ns()
                func()
                t1 = time.perf_counter_ns()
                timings.append((t1 - t0) / 1e9)
        freq_tup = psutil.cpu_freq()
        cpu_hz = (
            freq_tup.current if freq_tup and freq_tup.current else 0.0
        ) * 1e6
    finally:
        if gc_was_enabled:
            gc.enable()

    runs = len(timings)
    # core stats
    min_time = min(timings)
    max_time = max(timings)
    mean_time = statistics.mean(timings)
    try:
        mode_time = statistics.mode(timings)
    except statistics.StatisticsError:
        mode_time = float("nan")
    median_time = statistics.median(timings)
    std_time = statistics.stdev(timings) if runs > 1 else 0.0

    # throughput & cycle/flop estimates on best-case (min_time)
    throughput = 1.0 / min_time
    cycles_per_call = min_time * cpu_hz
    flops_per_call = ops_per_call
    flops_per_cycle = (
        ops_per_call / cycles_per_call if cycles_per_call else float("nan")
    )

    # build report
    labels = [
        ("Throughput      (calls/s)", f"{throughput:,.0f}"),
        ("CPU cycles      (cycles/call)", f"{cycles_per_call:,.0f}"),
        ("Est. Flops      (flops/call)", f"{flops_per_call:,.0f}"),
        ("Efficiency      (flops/cycle)", f"{flops_per_cycle:.4f}"),
        ("Min time        (ms/call)", f"{min_time*1e3:.3f}"),
        ("STDV time       (ms/call)", f"±{std_time*1e3:.3f}"),
    ]
    header = f"Benchmark: {runs} function calls."
    lines = [header] + [f"  {label:<30}{value}" for label, value in labels]
    report = "\n".join(lines)

    array_stats = [
        runs,
        min_time,
        max_time,
        mean_time,
        mode_time,
        median_time,
        std_time,
        throughput,
        cycles_per_call,
        flops_per_call,
        flops_per_cycle,
    ]
    array_headers = [
        "function_calls",
        "min_time_s",
        "max_time_s",
        "mean_time_s",
        "mode_time_s",
        "median_time_s",
        "std_time_s",
        "throughput_calls_per_s",
        "cycles_per_call",
        "flops_per_call",
        "flops_per_cycle",
    ]

    return {
        "min_time_s": min_time,
        "max_time_s": max_time,
        "mean_time_s": mean_time,
        "mode_time_s": mode_time,
        "median_time_s": median_time,
        "std_time_s": std_time,
        "throughput_calls_per_s": throughput,
        "cycles_per_call": cycles_per_call,
        "flops_per_call": flops_per_call,
        "flops_per_cycle": flops_per_cycle,
        "function_calls": runs,
        "report": report,
        "array_stats": array_stats,
        "array_headers": array_headers,
    }


def save_combined_csv(csv_path: str, header: list[str], *result_matrices):
    """
    Write one or more 2-D `result_matrices` into a single CSV file.

    Each matrix is an iterable of rows; every row length must match
    `header`.  Matrices are concatenated in the order provided.
    """
    rows = [row for matrix in result_matrices for row in matrix]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
