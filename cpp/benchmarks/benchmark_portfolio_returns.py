"""Reproducible NumPy-versus-C++ weighted-return microbenchmark."""

import argparse
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from statistics import median

import numpy as np

from riskchallenger.cpp_acceleration import weighted_portfolio_returns_cpp

ROOT = Path(__file__).resolve().parents[2]


def compile_executable(compiler: str, output_path: Path) -> None:
    """Build an optimized standalone executable for this benchmark invocation."""
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O3",
            "-I",
            str(ROOT / "cpp" / "include"),
            str(ROOT / "cpp" / "src" / "portfolio_returns.cpp"),
            str(ROOT / "cpp" / "src" / "portfolio_returns_cli.cpp"),
            "-o",
            str(output_path),
        ],
        check=True,
    )


def median_runtime(function, repetitions: int) -> float:
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--assets", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    if min(args.rows, args.assets, args.repetitions) <= 0:
        parser.error("rows, assets, and repetitions must be positive")
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        parser.error("benchmark requires g++ or clang++")

    generator = np.random.default_rng(args.seed)
    returns = np.ascontiguousarray(
        generator.normal(0.0, 0.01, size=(args.rows, args.assets)), dtype=np.float64
    )
    weights = generator.random(args.assets)
    weights /= weights.sum()

    with tempfile.TemporaryDirectory(prefix="riskchallenger_cpp_") as temp_dir:
        executable_path = Path(temp_dir) / (
            "riskchallenger_cpp.exe" if os.name == "nt" else "riskchallenger_cpp"
        )
        compile_executable(compiler, executable_path)

        def native():
            return weighted_portfolio_returns_cpp(returns, weights, executable_path)

        def python():
            return returns @ weights

        expected = python()
        actual = native()
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
        python_seconds = median_runtime(python, args.repetitions)
        cpp_seconds = median_runtime(native, args.repetitions)

    print(f"Compiler: {compiler}")
    processor = platform.processor() or "processor unspecified"
    print(f"Environment: {platform.platform()} | {processor}")
    print(f"NumPy: {np.__version__}; threads are governed by the active BLAS runtime")
    print(f"Input: {args.rows:,} observations x {args.assets} assets")
    print(f"Repetitions: {args.repetitions}; median; compilation excluded")
    print(f"Python NumPy median: {python_seconds:.6f} s")
    print(f"C++ subprocess median: {cpp_seconds:.6f} s")
    print(f"Speedup (Python / C++): {python_seconds / cpp_seconds:.3f}x")
    print("Numerical comparison: passed (rtol=1e-12, atol=1e-14)")


if __name__ == "__main__":
    main()
