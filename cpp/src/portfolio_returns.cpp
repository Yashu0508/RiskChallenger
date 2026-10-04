#include "portfolio_returns.hpp"

#include <cmath>

extern "C" int riskchallenger_weighted_returns(
    const double* row_major_returns,
    const std::size_t rows,
    const std::size_t assets,
    const double* weights,
    double* output
) {
    if (row_major_returns == nullptr || weights == nullptr || output == nullptr ||
        rows == 0 || assets == 0) {
        return PORTFOLIO_RETURNS_INVALID_ARGUMENT;
    }

    for (std::size_t asset = 0; asset < assets; ++asset) {
        if (!std::isfinite(weights[asset])) {
            return PORTFOLIO_RETURNS_NONFINITE_INPUT;
        }
    }

    for (std::size_t row = 0; row < rows; ++row) {
        double total = 0.0;
        const std::size_t offset = row * assets;
        for (std::size_t asset = 0; asset < assets; ++asset) {
            const double observation = row_major_returns[offset + asset];
            if (!std::isfinite(observation)) {
                return PORTFOLIO_RETURNS_NONFINITE_INPUT;
            }
            total += observation * weights[asset];
        }
        if (!std::isfinite(total)) {
            return PORTFOLIO_RETURNS_NONFINITE_OUTPUT;
        }
        output[row] = total;
    }
    return PORTFOLIO_RETURNS_OK;
}
