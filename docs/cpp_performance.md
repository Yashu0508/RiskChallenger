# Optional C++ performance layer

## Purpose and scope

Phase 10 adds a small C++17 weighted portfolio-return kernel as a performance
engineering experiment. Python remains the primary application and default
calculation path. The dashboard, API, and Phase 2 portfolio engine do not depend
on a compiled artifact, and the core engine was not changed.

The chosen operation is the row-wise portfolio return dot product over a
row-major matrix of decimal asset returns:

\[
R_{p,t}=\sum_{i=1}^{N} w_i r_{i,t}
\]

It returns decimal portfolio returns (for example, `0.01` means 1%). It does
not calculate P&L, evolve portfolio value, normalize weights, or handle dates;
those remain the established Python layers' responsibilities. Inputs must be
finite, non-empty arrays, and the weight count must match the matrix's asset
dimension. Portfolio weight policy and fully-invested validation remain in
Phase 2.

## Architecture and Python interface

`cpp/src/portfolio_returns.cpp` implements the computation behind a small C
ABI. `portfolio_returns_cli.cpp` exposes that function as a standalone
executable. It accepts binary stdin containing two native-endian `uint64`
dimensions, row-major `float64` returns, then `float64` weights; stdout contains
one `float64` result per row. The CLI explicitly switches Windows stdin/stdout
to binary mode.

Python's optional
[`weighted_portfolio_returns_cpp`](../src/riskchallenger/cpp_acceleration.py)
validates and serializes the arrays, launches the executable, and checks its
exit status and output length. Because the available MinGW compiler is 32-bit
(`mingw32`) while the project's Python is 64-bit, its generated shared library
cannot load into this Python process. The standalone-process interface avoids
that ABI mismatch and adds no Python dependency. Python integration is
available through the executable, but the Phase 2 engine intentionally does
not select it automatically.

## Build and test

From the repository root on the validated Windows/MinGW environment:

```powershell
New-Item -ItemType Directory -Force cpp/build
g++ -std=c++17 -O3 -Wall -Wextra -Werror -I cpp/include `
  cpp/src/portfolio_returns.cpp cpp/src/portfolio_returns_cli.cpp `
  -o cpp/build/riskchallenger_cpp.exe
g++ -std=c++17 -O2 -Wall -Wextra -Werror -I cpp/include `
  cpp/src/portfolio_returns.cpp cpp/tests/test_portfolio_returns.cpp `
  -o cpp/build/test_portfolio_returns.exe
cpp/build/test_portfolio_returns.exe
python -m pytest tests/test_cpp_acceleration.py -q
```

The C++ tests cover a known weighted result, zero returns, null pointers, and
non-finite inputs. The Python integration tests compile a temporary executable,
compare it against NumPy, test deterministic output, and reject invalid array
shapes and non-finite values. They skip when neither `g++` nor `clang++` is
available.

## Numerical validation

The convention matches the existing Phase 2 vector product. Deterministic
examples are checked against manually known values and NumPy using
`rtol=1e-12`, `atol=1e-14`. The kernel uses ordinary double-precision
accumulation in asset order; tiny floating-point differences from NumPy's
matrix multiplication order are expected and remain within these tolerances.

## Controlled benchmark

Run the reproducible synthetic benchmark from the repository root:

```powershell
python cpp/benchmarks/benchmark_portfolio_returns.py `
  --rows 200000 --assets 50 --repetitions 7 --seed 20261004
```

It generates fixed-seed Gaussian decimal returns, uses a normalized synthetic
weight vector, verifies numerical agreement before timing, excludes compilation,
and reports median runtimes. The NumPy measurement is `returns @ weights`; the
C++ measurement includes the Python adapter's serialization, process launch,
native calculation, and returned-output transfer. This end-to-end convention
intentionally reflects the actual standalone integration boundary. BLAS thread
settings are inherited from the active NumPy environment.

On the validation machine (Windows 11, Intel64 Family 6 Model 154, Python
64-bit, NumPy 2.2.1, MinGW.org GCC 6.3.0 `mingw32`), for 200,000 observations ×
50 assets over 7 repetitions, median NumPy time was **0.004216 s** and median
Python-to-C++ subprocess time was **0.151036 s**. That is **0.028× speedup**
(equivalently, the subprocess path took about 35.8× as long). Compilation was
excluded. The measured integration is not faster, so it is not wired into the
production portfolio calculation; the result demonstrates the cost of
cross-process serialization for this operation and workload. These timings
are environment-specific, not a general C++ versus NumPy claim.

## Limitations

- The C++ kernel is optional and currently not used by the dashboard/API or
  default Phase 2 workflow.
- Python's optimized NumPy/BLAS implementation is substantially faster for the
  measured matrix shape once process-boundary costs are included.
- The standalone protocol is intended for same-machine use; it is not a
  network protocol or a stable cross-endian file format.
- It does not accelerate risk fitting, quantile/ES estimation, Student-t fitting,
  or GARCH. Those validated statistical implementations remain in Python.
