"""
FREAK (Fast Retina Keypoint) descriptor implementation

Based on MindAR's JavaScript implementation for compatibility.
"""

import math
from typing import List, Tuple

import numpy as np

# MindAR compatible FREAK pattern (37 points)
FREAK_RINGS = [
    # ring 5
    {
        "sigma": 0.550000,
        "points": [
            [-1.000000, 0.000000],
            [-0.500000, -0.866025],
            [0.500000, -0.866025],
            [1.000000, -0.000000],
            [0.500000, 0.866025],
            [-0.500000, 0.866025],
        ],
    },
    # ring 4
    {
        "sigma": 0.475000,
        "points": [
            [0.000000, 0.930969],
            [-0.806243, 0.465485],
            [-0.806243, -0.465485],
            [-0.000000, -0.930969],
            [0.806243, -0.465485],
            [0.806243, 0.465485],
        ],
    },
    # ring 3
    {
        "sigma": 0.400000,
        "points": [
            [0.847306, -0.000000],
            [0.423653, 0.733789],
            [-0.423653, 0.733789],
            [-0.847306, 0.000000],
            [-0.423653, -0.733789],
            [0.423653, -0.733789],
        ],
    },
    # ring 2
    {
        "sigma": 0.325000,
        "points": [
            [-0.000000, -0.741094],
            [0.641806, -0.370547],
            [0.641806, 0.370547],
            [0.000000, 0.741094],
            [-0.641806, 0.370547],
            [-0.641806, -0.370547],
        ],
    },
    # ring 1
    {
        "sigma": 0.250000,
        "points": [
            [-0.595502, 0.000000],
            [-0.297751, -0.515720],
            [0.297751, -0.515720],
            [0.595502, -0.000000],
            [0.297751, 0.515720],
            [-0.297751, 0.515720],
        ],
    },
    # ring 0
    {
        "sigma": 0.175000,
        "points": [
            [0.000000, 0.362783],
            [-0.314179, 0.181391],
            [-0.314179, -0.181391],
            [-0.000000, -0.362783],
            [0.314179, -0.181391],
            [0.314179, 0.181391],
        ],
    },
    # center
    {"sigma": 0.100000, "points": [[0, 0]]},
]

# Build FREAK points list compatible with MindAR
FREAKPOINTS = []
for ring in FREAK_RINGS:
    sigma = ring["sigma"]
    for point in ring["points"]:
        FREAKPOINTS.append([sigma, point[0], point[1]])

# Comparison pairs count
FREAK_COMPARISON_COUNT = (len(FREAKPOINTS) - 1) * len(FREAKPOINTS) // 2
DESCRIPTOR_COUNT = math.ceil(FREAK_COMPARISON_COUNT / 8)


class FreakDescriptor:
    """
    MindAR-compatible FREAK descriptor implementation.
    """

    def __init__(self):
        """Initialize FREAK descriptor."""
        self.freakpoints = FREAKPOINTS
        self.comparison_count = FREAK_COMPARISON_COUNT
        self.descriptor_count = DESCRIPTOR_COUNT

    def compute_descriptors(
        self, image: np.ndarray, keypoints: List[Tuple[float, float, float, float]]
    ) -> List[List[int]]:
        """
        Compute FREAK descriptors for keypoints (MindAR compatible).

        Args:
            image: Input grayscale image
            keypoints: List of (x, y, scale, angle) tuples

        Returns:
            List of descriptor lists (integers)
        """
        if not keypoints:
            return []

        descriptors = []

        for x, y, scale, angle in keypoints:
            descriptor = self._compute_single_descriptor(image, x, y, scale, angle)
            descriptors.append(descriptor)

        return descriptors

    def _compute_single_descriptor(
        self, image: np.ndarray, x: float, y: float, scale: float, angle: float
    ) -> List[int]:
        """Compute FREAK descriptor for a single keypoint."""
        try:
            # Sample intensity values at FREAK pattern points
            pattern_responses = []

            cos_angle = math.cos(angle)
            sin_angle = math.sin(angle)

            for sigma, px, py in self.freakpoints:
                # Scale and rotate pattern point
                scaled_px = px * sigma * scale * 20  # Scale factor for visibility
                scaled_py = py * sigma * scale * 20

                # Rotate pattern point
                rotated_px = scaled_px * cos_angle - scaled_py * sin_angle
                rotated_py = scaled_px * sin_angle + scaled_py * cos_angle

                # Sample location in image
                sample_x = x + rotated_px
                sample_y = y + rotated_py

                # Sample intensity with bounds checking
                intensity = self._sample_intensity(image, sample_x, sample_y)
                pattern_responses.append(intensity)

            # Generate binary descriptor by comparing pairs
            descriptor_ints = []
            comparison_idx = 0

            for desc_byte_idx in range(self.descriptor_count):
                byte_value = 0

                for bit_idx in range(8):
                    if comparison_idx >= self.comparison_count:
                        break

                    # Get pair indices for this comparison
                    p1_idx, p2_idx = self._get_comparison_pair(comparison_idx)

                    if p1_idx < len(pattern_responses) and p2_idx < len(pattern_responses):
                        # Compare intensities (MindAR format: < comparison)
                        if pattern_responses[p1_idx] < pattern_responses[p2_idx] + 0.01:
                            byte_value += int(2 ** (7 - bit_idx))

                    comparison_idx += 1

                descriptor_ints.append(byte_value)

            return descriptor_ints

        except Exception:
            # Return empty descriptor on error
            return [0] * self.descriptor_count

    def _get_comparison_pair(self, comparison_idx: int) -> Tuple[int, int]:
        """Get the pair of point indices for a given comparison index."""
        # Generate pairs systematically as in MindAR
        count = 0
        for i in range(len(self.freakpoints)):
            for j in range(i + 1, len(self.freakpoints)):
                if count == comparison_idx:
                    return i, j
                count += 1
        return 0, 1  # Fallback

    def _sample_intensity(self, image: np.ndarray, x: float, y: float) -> float:
        """Sample intensity with bounds checking."""
        x_int = int(round(x))
        y_int = int(round(y))

        # Check bounds
        if x_int < 0 or x_int >= image.shape[1] or y_int < 0 or y_int >= image.shape[0]:
            return 0.0

        return float(image[y_int, x_int])
