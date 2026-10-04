#include "portfolio_returns.hpp"

#include <cmath>
#include <iostream>
#include <limits>

namespace {
bool close(const double left, const double right, const double tolerance = 1e-14) {
    return std::abs(left - right) <= tolerance;
}
}  // namespace

int main() {
    const double returns[] = {0.10, 0.00, -0.02, 0.04, 0.01, -0.03};
    const double weights[] = {0.60, 0.40, 0.00};
    double output[2] = {};
    if (riskchallenger_weighted_returns(returns, 2, 3, weights, output) !=
            PORTFOLIO_RETURNS_OK ||
        !close(output[0], 0.06) || !close(output[1], 0.028)) {
        std::cerr << "known weighted-return example failed\n";
        return 1;
    }

    const double zero[] = {0.0, 0.0};
    const double one_weight[] = {0.25, 0.75};
    double zero_output[1] = {1.0};
    if (riskchallenger_weighted_returns(zero, 1, 2, one_weight, zero_output) !=
            PORTFOLIO_RETURNS_OK ||
        zero_output[0] != 0.0) {
        std::cerr << "zero-return edge case failed\n";
        return 1;
    }

    if (riskchallenger_weighted_returns(nullptr, 1, 2, one_weight, zero_output) !=
        PORTFOLIO_RETURNS_INVALID_ARGUMENT) {
        std::cerr << "null-pointer validation failed\n";
        return 1;
    }

    const double invalid[] = {std::numeric_limits<double>::infinity(), 0.0};
    if (riskchallenger_weighted_returns(invalid, 1, 2, one_weight, zero_output) !=
        PORTFOLIO_RETURNS_NONFINITE_INPUT) {
        std::cerr << "non-finite input validation failed\n";
        return 1;
    }

    std::cout << "C++ portfolio-return tests passed\n";
    return 0;
}
