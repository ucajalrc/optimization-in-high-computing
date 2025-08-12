"""
Benchmark: Eliminating Inefficient Algorithm Implementations (HPC Optimization Demo)

What this script does
---------------------
- Compares a naive O(n^2) frequency counter with an optimized O(n) version.
- Benchmarks across multiple dataset sizes with repeated trials.
- Controls sources of noise (warm-up, GC, stable RNG).
- Validates that both implementations produce identical results.
- Prints a clean summary table, saves results to CSV, and (optionally) plots.

How to run
----------
$ python benchmark_freq.py

Optional: to enable plots, ensure matplotlib is installed:
$ pip install matplotlib
"""

from __future__ import annotations

import gc
import math
import random
import statistics as stats
import time
from typing import Callable, Dict, Iterable, List, Tuple

# -------------------------
# Implementations under test
# -------------------------

def freq_naive(data: List[int]) -> Dict[int, int]:
    """
    Naive frequency counter using list.count() inside a loop.

    Complexity:
        O(n^2) because data.count(x) is O(n) and we do it for each element.
    Notes:
        This is intentionally inefficient to illustrate the impact of algorithmic choice.
    """
    result: Dict[int, int] = {}
    for x in data:
        # Re-counts the entire list for each x -> quadratic time
        result[x] = data.count(x)
    return result


def freq_fast(data: List[int]) -> Dict[int, int]:
    """
    Optimized frequency counter using a dictionary (hash table).

    Complexity:
        O(n) average-case: we make a single pass and do O(1)-amortized updates.
    """
    result: Dict[int, int] = {}
    for x in data:
        result[x] = result.get(x, 0) + 1
    return result


# -------------------------
# Benchmarking utilities
# -------------------------

def make_dataset(n: int, domain: int = 1000, seed: int | None = None) -> List[int]:
    """
    Create a synthetic dataset of length n with integers in [0, domain].

    Args:
        n: number of elements to generate
        domain: value range to draw from (smaller domain => more repeats)
        seed: optional RNG seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
    return [random.randint(0, domain) for _ in range(n)]


def time_once(fn: Callable[[List[int]], Dict[int, int]],
              data: List[int]) -> Tuple[float, Dict[int, int]]:
    """
    Time a single execution of `fn(data)` using perf_counter.

    Returns:
        (elapsed_seconds, output_dict)
    """
    # Encourage a clean run each time
    gc.collect()
    start = time.perf_counter()
    out = fn(data)
    elapsed = time.perf_counter() - start
    return elapsed, out


def repeat_timing(fn: Callable[[List[int]], Dict[int, int]],
                  data: List[int],
                  repeats: int = 5,
                  warmup: int = 1) -> Tuple[List[float], Dict[int, int]]:
    """
    Time `fn(data)` multiple times with warm-ups and return all durations.

    Args:
        fn: function to benchmark
        data: input list
        repeats: number of measured runs
        warmup: number of unmeasured warm-up runs to 'prime' caches/JITs

    Returns:
        (list_of_durations, last_output)
    """
    # Warm-up runs (unmeasured)
    for _ in range(warmup):
        _ = fn(data)

    durations: List[float] = []
    last_out: Dict[int, int] = {}
    for _ in range(repeats):
        t, out = time_once(fn, data)
        durations.append(t)
        last_out = out
    return durations, last_out


def summarize(durations: List[float]) -> Tuple[float, float, float]:
    """
    Return (min, mean, std) for a list of durations.
    """
    if len(durations) == 1:
        return durations[0], durations[0], 0.0
    return min(durations), stats.mean(durations), stats.pstdev(durations)


# -------------------------
# Main experiment
# -------------------------

def run_experiment(
    sizes: Iterable[int] = (10_000, 20_000, 50_000, 80_000, 100_000),
    repeats: int = 5,
    warmup: int = 1,
    domain: int = 1000,
    seed: int = 42,
    save_csv_path: str = "benchmark_results.csv",
    make_plot: bool = True,
) -> None:
    """
    Execute the full benchmark across dataset sizes.

    Prints a table with (min/mean/std) for each method and their speedup,
    saves a CSV, and optionally plots results.

    Args:
        sizes: dataset sizes to test
        repeats: measured runs per size per method
        warmup: unmeasured warm-up runs
        domain: value range for synthetic data
        seed: RNG seed for reproducibility
        save_csv_path: output CSV filename
        make_plot: whether to render a matplotlib plot
    """
    # Late import so the script runs even if matplotlib isn't installed
    if make_plot:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            print("[info] matplotlib not found; continuing without plots.")
            make_plot = False

    # Print table header
    header = (
        "n, "
        "naive_min_s, naive_mean_s, naive_std_s, "
        "fast_min_s, fast_mean_s, fast_std_s, "
        "speedup_mean_x"
    )
    print(header)

    rows_for_csv: List[str] = [header]

    # Iterate each size and benchmark both functions
    for n in sizes:
        # Fresh dataset per size (controlled RNG for reproducibility)
        data = make_dataset(n=n, domain=domain, seed=seed)

        # --- Naive ---
        naive_durations, naive_out = repeat_timing(freq_naive, data, repeats=repeats, warmup=warmup)
        naive_min, naive_mean, naive_std = summarize(naive_durations)

        # --- Optimized ---
        fast_durations, fast_out = repeat_timing(freq_fast, data, repeats=repeats, warmup=warmup)
        fast_min, fast_mean, fast_std = summarize(fast_durations)

        # --- Correctness check ---
        assert naive_out == fast_out, (
            f"Output mismatch at n={n}: results differ between naive and fast implementations."
        )

        # --- Speedup (using mean-to-mean for stability) ---
        speedup_mean = naive_mean / fast_mean if fast_mean > 0 else math.inf

        # Print row
        row = (
            f"{n}, "
            f"{naive_min:.6f}, {naive_mean:.6f}, {naive_std:.6f}, "
            f"{fast_min:.6f}, {fast_mean:.6f}, {fast_std:.6f}, "
            f"{speedup_mean:.2f}"
        )
        print(row)
        rows_for_csv.append(row)

    # Save CSV
    with open(save_csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows_for_csv))
    print(f"\n[ok] Saved CSV to: {save_csv_path}")

    # Optional plotting
    if make_plot:
        # Re-parse rows for plotting (avoid capturing inside main loop to keep it simple)
        ns: List[int] = []
        naive_means: List[float] = []
        fast_means: List[float] = []
        speedups: List[float] = []

        for i, line in enumerate(rows_for_csv):
            if i == 0:  # skip header
                continue
            parts = [p.strip() for p in line.split(",")]
            ns.append(int(parts[0]))
            naive_means.append(float(parts[2]))
            fast_means.append(float(parts[5]))
            speedups.append(float(parts[7]))

        # Plot runtimes
        import matplotlib.pyplot as plt  # type: ignore

        plt.figure()
        plt.plot(ns, naive_means, marker="o", label="Naive (mean s)")
        plt.plot(ns, fast_means, marker="o", label="Optimized (mean s)")
        plt.xlabel("Dataset size (n)")
        plt.ylabel("Time (seconds)")
        plt.title("Runtime vs. n (mean over repeats)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("runtime_vs_n.png", dpi=160)
        print("[ok] Saved plot: runtime_vs_n.png")

        # Plot speedup
        plt.figure()
        plt.plot(ns, speedups, marker="o", label="Speedup (naive_mean / fast_mean)")
        plt.xlabel("Dataset size (n)")
        plt.ylabel("Speedup (×)")
        plt.title("Speedup vs. n")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("speedup_vs_n.png", dpi=160)
        print("[ok] Saved plot: speedup_vs_n.png")


# -------------------------
# Script entry point
# -------------------------

if __name__ == "__main__":
    # You can tweak sizes/repeats below if you want a faster run.
    run_experiment(
        sizes=(10_000, 20_000, 50_000, 80_000, 100_000),
        repeats=5,          # increase to 10+ for even stabler stats
        warmup=1,           # 1–3 is typical
        domain=1000,        # smaller domain -> more duplicates -> realistic counting
        seed=1337,          # fixed seed for reproducibility
        save_csv_path="benchmark_results.csv",
        make_plot=True,     # set False if you don't want PNG outputs
    )
