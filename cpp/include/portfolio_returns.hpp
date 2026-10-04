#ifndef RISKCHALLENGER_PORTFOLIO_RETURNS_HPP
#define RISKCHALLENGER_PORTFOLIO_RETURNS_HPP

#include <cstddef>
#include <cstdint>

// Return codes for the stable C ABI used by both C++ tests and ctypes.
enum PortfolioReturnsStatus {
    PORTFOLIO_RETURNS_OK = 0,
    PORTFOLIO_RETURNS_INVALID_ARGUMENT = 1,
    PORTFOLIO_RETURNS_NONFINITE_INPUT = 2,
    PORTFOLIO_RETURNS_NONFINITE_OUTPUT = 3
};

extern "C" int riskchallenger_weighted_returns(
    const double* row_major_returns,
    std::size_t rows,
    std::size_t assets,
    const double* weights,
    double* output
);

int run_portfolio_returns_cli();

#endif
