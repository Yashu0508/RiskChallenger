#include "portfolio_returns.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

int run_portfolio_returns_cli() {
#if defined(_WIN32)
    _setmode(0, _O_BINARY);
    _setmode(1, _O_BINARY);
#endif
    std::uint64_t rows_wire = 0;
    std::uint64_t assets_wire = 0;
    std::cin.read(reinterpret_cast<char*>(&rows_wire), sizeof(rows_wire));
    std::cin.read(reinterpret_cast<char*>(&assets_wire), sizeof(assets_wire));
    if (!std::cin || rows_wire == 0 || assets_wire == 0 ||
        rows_wire > std::numeric_limits<std::size_t>::max() ||
        assets_wire > std::numeric_limits<std::size_t>::max() ||
        rows_wire > std::numeric_limits<std::size_t>::max() / assets_wire) {
        std::cerr << "invalid dimensions in binary request\n";
        return 2;
    }

    const auto rows = static_cast<std::size_t>(rows_wire);
    const auto assets = static_cast<std::size_t>(assets_wire);
    std::vector<double> observations(rows * assets);
    std::vector<double> weights(assets);
    std::vector<double> output(rows);
    std::cin.read(
        reinterpret_cast<char*>(observations.data()),
        static_cast<std::streamsize>(observations.size() * sizeof(double))
    );
    std::cin.read(
        reinterpret_cast<char*>(weights.data()),
        static_cast<std::streamsize>(weights.size() * sizeof(double))
    );
    if (!std::cin) {
        std::cerr << "truncated binary request\n";
        return 2;
    }

    const int status = riskchallenger_weighted_returns(
        observations.data(), rows, assets, weights.data(), output.data()
    );
    if (status != PORTFOLIO_RETURNS_OK) {
        std::cerr << "portfolio-return kernel rejected input (status " << status
                  << ")\n";
        return 3;
    }
    std::cout.write(
        reinterpret_cast<const char*>(output.data()),
        static_cast<std::streamsize>(output.size() * sizeof(double))
    );
    return std::cout ? 0 : 4;
}

int main() { return run_portfolio_returns_cli(); }
