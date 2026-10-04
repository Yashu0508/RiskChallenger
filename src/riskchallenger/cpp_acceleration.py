"""Optional subprocess adapter for the standalone C++ weighted-return kernel.

The core portfolio engine remains NumPy-based. The adapter communicates with
the optional C++17 executable using a compact binary protocol over stdin/stdout.
"""

import struct
import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


def weighted_portfolio_returns_cpp(
    asset_returns: ArrayLike,
    weights: ArrayLike,
    executable_path: str | Path,
) -> NDArray[np.float64]:
    """Calculate row-wise ``asset_returns @ weights`` using the C++ executable.

    Returns decimal portfolio returns. Inputs are converted to contiguous
    float64 arrays, but values are never rounded, normalized, or imputed.
    Process startup and binary serialization are part of this optional API's
    runtime and should be included in end-to-end performance measurements.
    """
    returns = np.asarray(asset_returns, dtype=np.float64)
    weight_vector = np.asarray(weights, dtype=np.float64)
    if returns.ndim != 2 or returns.shape[0] == 0 or returns.shape[1] == 0:
        raise ValueError("asset_returns must be a non-empty 2D array")
    if weight_vector.ndim != 1 or weight_vector.size != returns.shape[1]:
        raise ValueError("weights must be 1D and match the return asset dimension")
    if not np.isfinite(returns).all() or not np.isfinite(weight_vector).all():
        raise ValueError("returns and weights must contain only finite values")

    returns = np.ascontiguousarray(returns)
    weight_vector = np.ascontiguousarray(weight_vector)
    request = (
        struct.pack("=QQ", returns.shape[0], returns.shape[1])
        + returns.tobytes(order="C")
        + weight_vector.tobytes(order="C")
    )
    executable = Path(executable_path).resolve()
    try:
        process = subprocess.run(
            [str(executable)],
            input=request,
            capture_output=True,
            check=False,
        )
    except OSError as error:
        raise RuntimeError(f"Could not run C++ executable: {executable}") from error
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"C++ portfolio-return executable failed: {detail}")
    expected_bytes = returns.shape[0] * np.dtype(np.float64).itemsize
    if len(process.stdout) != expected_bytes:
        raise RuntimeError("C++ executable returned an invalid output length")
    return np.frombuffer(process.stdout, dtype=np.float64).copy()
