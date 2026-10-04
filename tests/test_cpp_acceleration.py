"""Optional Python/native parity tests, skipped without an available compiler."""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from riskchallenger.cpp_acceleration import weighted_portfolio_returns_cpp

ROOT = Path(__file__).resolve().parents[1]
COMPILER = shutil.which("g++") or shutil.which("clang++")


@pytest.fixture(scope="module")
def native_executable(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if COMPILER is None:
        pytest.skip("C++ integration test requires g++ or clang++")
    output = tmp_path_factory.mktemp("native") / (
        "riskchallenger_cpp.exe" if os.name == "nt" else "riskchallenger_cpp"
    )
    subprocess.run(
        [
            COMPILER,
            "-std=c++17",
            "-O3",
            "-I",
            str(ROOT / "cpp" / "include"),
            str(ROOT / "cpp" / "src" / "portfolio_returns.cpp"),
            str(ROOT / "cpp" / "src" / "portfolio_returns_cli.cpp"),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return output


def test_cpp_matches_numpy_and_known_value(native_executable: Path) -> None:
    returns = np.array([[0.10, 0.00], [0.0, 0.05], [-0.1, 0.02]])
    weights = np.array([0.6, 0.4])
    native = weighted_portfolio_returns_cpp(returns, weights, native_executable)
    expected = returns @ weights
    np.testing.assert_allclose(native, expected, rtol=1e-12, atol=1e-14)
    assert native[0] == pytest.approx(0.06)


def test_cpp_wrapper_rejects_invalid_shapes_and_nonfinite_values(
    native_executable: Path,
) -> None:
    with pytest.raises(ValueError, match="non-empty 2D"):
        weighted_portfolio_returns_cpp(np.empty((0, 2)), [0.5, 0.5], native_executable)
    with pytest.raises(ValueError, match="match the return asset dimension"):
        weighted_portfolio_returns_cpp([[0.1, 0.2]], [1.0], native_executable)
    with pytest.raises(ValueError, match="finite"):
        weighted_portfolio_returns_cpp([[np.nan]], [1.0], native_executable)


def test_cpp_wrapper_zero_returns_are_deterministic(native_executable: Path) -> None:
    returns = np.zeros((10, 3))
    weights = np.array([0.2, 0.3, 0.5])
    first = weighted_portfolio_returns_cpp(returns, weights, native_executable)
    second = weighted_portfolio_returns_cpp(returns, weights, native_executable)
    np.testing.assert_array_equal(first, np.zeros(10))
    np.testing.assert_array_equal(first, second)
