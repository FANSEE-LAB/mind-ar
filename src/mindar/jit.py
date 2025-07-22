"""
Unified JIT compilation management module.

Provides conditional numba import, precompiled functions, and graceful fallback mechanism.
Ensures normal operation even without numba.
"""

import warnings

# Dynamically check numba availability
try:
    import numba
    from numba import jit

    NUMBA_AVAILABLE = True
    NUMBA_VERSION = getattr(numba, "__version__", "unknown")
except ImportError:
    NUMBA_AVAILABLE = False
    NUMBA_VERSION = None

    # Create a no-op jit decorator
    def jit(*args, **kwargs):
        """Fallback jit decorator that does nothing when numba is not available."""

        def decorator(func):
            return func

        return decorator


def warn_fallback(function_name: str) -> None:
    """Warn user that fallback implementation is used."""
    if not NUMBA_AVAILABLE:
        warnings.warn(
            f"numba is not available, {function_name} uses pure Python implementation. "
            f"For best performance, please install numba: pip install 'mindar[performance]'",
            UserWarning,
            stacklevel=3,
        )


def get_jit_info() -> dict:
    """
    Get JIT compilation environment info.

    Returns:
        Dictionary containing JIT status
    """
    return {"numba_available": NUMBA_AVAILABLE, "numba_version": NUMBA_VERSION, "jit_enabled": NUMBA_AVAILABLE}


# Core computation functions


def _hamming_distance_impl(desc1, desc2):
    """Core implementation of Hamming distance calculation."""
    distance = 0
    min_len = min(len(desc1), len(desc2))
    for i in range(min_len):
        # Ensure all operations are on integers
        val1 = int(desc1[i])
        val2 = int(desc2[i])
        # Bitwise XOR
        xor_result = val1 ^ val2

        # Brian Kernighan's algorithm to count bits
        # Ensure all variables are integers
        while xor_result > 0:
            distance += 1
            xor_result = xor_result & (xor_result - 1)
    return distance


def _harris_response_impl(image, x_coord, y_coord, window_size, harris_k):
    """Core implementation of Harris corner response calculation."""
    if (
        x_coord < window_size
        or x_coord >= image.shape[1] - window_size
        or y_coord < window_size
        or y_coord >= image.shape[0] - window_size
    ):
        return 0.0

    grad_xx = 0.0
    grad_yy = 0.0
    grad_xy = 0.0
    half_window = window_size // 2

    for delta_y in range(-half_window, half_window + 1):
        for delta_x in range(-half_window, half_window + 1):
            y_pos = y_coord + delta_y
            x_pos = x_coord + delta_x

            # Compute gradients
            grad_x = (float(image[y_pos, x_pos + 1]) - float(image[y_pos, x_pos - 1])) * 0.5
            grad_y = (float(image[y_pos + 1, x_pos]) - float(image[y_pos - 1, x_pos])) * 0.5

            grad_xx += grad_x * grad_x
            grad_yy += grad_y * grad_y
            grad_xy += grad_x * grad_y

    # Harris corner response
    det = grad_xx * grad_yy - grad_xy * grad_xy
    trace = grad_xx + grad_yy

    if trace == 0.0:
        return 0.0

    return det - harris_k * (trace * trace)


# Create final functions based on numba availability
if NUMBA_AVAILABLE:
    # JIT-compiled version - use lazy compilation to avoid type issues
    _hamming_distance_jit_impl = jit(nopython=True, cache=True)(_hamming_distance_impl)
    _harris_response_jit_impl = jit(nopython=True, cache=True)(_harris_response_impl)
else:
    # Pure Python version
    _hamming_distance_jit_impl = _hamming_distance_impl
    _harris_response_jit_impl = _harris_response_impl
