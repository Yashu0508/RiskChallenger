# Optional C++ portfolio-return kernel

This directory contains an optional C++17 implementation of the per-row
weighted portfolio-return calculation. Python remains the default application
and calculation layer. See [`docs/cpp_performance.md`](../docs/cpp_performance.md)
for mathematical conventions, build commands, Python integration, tests, and
the reproducible benchmark.

Build the standalone executable and C++ test executable from the repository
root with MinGW `g++`:

```powershell
New-Item -ItemType Directory -Force cpp/build
g++ -std=c++17 -O3 -Wall -Wextra -Werror -I cpp/include `
  cpp/src/portfolio_returns.cpp cpp/src/portfolio_returns_cli.cpp `
  -o cpp/build/riskchallenger_cpp.exe
g++ -std=c++17 -O2 -Wall -Wextra -Werror -I cpp/include `
  cpp/src/portfolio_returns.cpp cpp/tests/test_portfolio_returns.cpp `
  -o cpp/build/test_portfolio_returns.exe
cpp/build/test_portfolio_returns.exe
```

The dashboard and API do not require this executable to be built.

