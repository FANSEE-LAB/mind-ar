"""
Optimized kernels for image processing operations

This module provides optimized implementations of key image processing operations
needed for feature detection and tracking. It includes both NumPy and optional
TensorFlow implementations for better performance on different hardware.
"""

import math
from typing import List, Tuple

import cv2
import numpy as np


def binomial_filter_1d(image: np.ndarray, horizontal: bool = True) -> np.ndarray:
    """
    Apply 1D binomial filter [1,4,6,4,1]/16 to image

    Args:
        image: Input image
        horizontal: If True, apply horizontally, else vertically

    Returns:
        Filtered image
    """
    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0

    if horizontal:
        # Apply horizontally
        padded = np.pad(image, ((0, 0), (2, 2)), mode="reflect")
        result = np.zeros_like(image)

        for i in range(image.shape[1]):
            result[:, i] = np.sum(padded[:, i : i + 5] * kernel, axis=1)
    else:
        # Apply vertically
        padded = np.pad(image, ((2, 2), (0, 0)), mode="reflect")
        result = np.zeros_like(image)

        for i in range(image.shape[0]):
            result[i, :] = np.sum(padded[i : i + 5, :] * kernel[:, np.newaxis], axis=0)

    return result


def binomial_filter_2d(image: np.ndarray) -> np.ndarray:
    """
    Apply 2D binomial filter to image (separable implementation)

    Args:
        image: Input image

    Returns:
        Filtered image
    """
    # Apply horizontal pass
    temp = binomial_filter_1d(image, horizontal=True)

    # Apply vertical pass
    return binomial_filter_1d(temp, horizontal=False)


def downsample_bilinear(image: np.ndarray) -> np.ndarray:
    """
    Downsample image by factor of 2 using bilinear interpolation

    Args:
        image: Input image

    Returns:
        Downsampled image
    """
    height, width = image.shape
    new_height, new_width = height // 2, width // 2

    result = np.zeros((new_height, new_width), dtype=np.float32)

    # Optimized implementation using 2x2 averaging
    for y in range(new_height):
        for x in range(new_width):
            y2, x2 = y * 2, x * 2
            result[y, x] = (image[y2, x2] + image[y2, x2 + 1] + image[y2 + 1, x2] + image[y2 + 1, x2 + 1]) / 4.0

    return result


def upsample_bilinear(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Upsample image using bilinear interpolation

    Args:
        image: Input image
        target_shape: Target shape (height, width)

    Returns:
        Upsampled image
    """
    src_height, src_width = image.shape
    tgt_height, tgt_width = target_shape

    result = np.zeros(target_shape, dtype=np.float32)

    # Scale factors
    scale_y = src_height / tgt_height
    scale_x = src_width / tgt_width

    for y in range(tgt_height):
        for x in range(tgt_width):
            # Source coordinates
            src_y = y * scale_y
            src_x = x * scale_x

            # Integer and fractional parts
            src_y0 = int(math.floor(src_y))
            src_x0 = int(math.floor(src_x))
            src_y1 = min(src_y0 + 1, src_height - 1)
            src_x1 = min(src_x0 + 1, src_width - 1)

            # Fractional part for interpolation
            fy = src_y - src_y0
            fx = src_x - src_x0

            # Bilinear interpolation
            top = image[src_y0, src_x0] * (1 - fx) + image[src_y0, src_x1] * fx
            bottom = image[src_y1, src_x0] * (1 - fx) + image[src_y1, src_x1] * fx
            result[y, x] = top * (1 - fy) + bottom * fy

    return result


def build_gaussian_pyramid(image: np.ndarray, num_octaves: int) -> List[List[np.ndarray]]:
    """
    Build Gaussian pyramid with specified number of octaves

    Args:
        image: Input image
        num_octaves: Number of octaves to build

    Returns:
        List of octaves, each containing two images (original and blurred)
    """
    pyramid = []
    current = image.copy()

    for i in range(num_octaves):
        # Create two images per octave
        image1 = current
        image2 = binomial_filter_2d(image1)
        pyramid.append([image1, image2])

        # Downsample for next octave
        if i < num_octaves - 1:
            current = downsample_bilinear(image2)

    return pyramid


def build_dog_pyramid(gaussian_pyramid: List[List[np.ndarray]]) -> List[np.ndarray]:
    """
    Build Difference of Gaussian pyramid from Gaussian pyramid

    Args:
        gaussian_pyramid: Gaussian pyramid from build_gaussian_pyramid

    Returns:
        List of DoG images
    """
    dog_pyramid = []

    for octave in gaussian_pyramid:
        # Compute difference between two images in octave
        dog = octave[0] - octave[1]
        dog_pyramid.append(dog)

    return dog_pyramid


def find_local_extrema(dog_pyramid: List[np.ndarray], threshold: float = 0.03) -> List[Tuple[int, int, int, bool]]:
    """
    Find local extrema in DoG pyramid

    Args:
        dog_pyramid: DoG pyramid from build_dog_pyramid
        threshold: Threshold for extrema detection

    Returns:
        List of (octave, y, x, is_maximum) tuples
    """
    extrema = []

    # For each octave except first and last
    for octave in range(1, len(dog_pyramid) - 1):
        # Get current, previous and next DoG images
        prev_dog = dog_pyramid[octave - 1]
        curr_dog = dog_pyramid[octave]
        next_dog = dog_pyramid[octave + 1]

        # Resize to match current octave
        prev_dog_resized = cv2.resize(prev_dog, (curr_dog.shape[1], curr_dog.shape[0]))
        next_dog_resized = cv2.resize(next_dog, (curr_dog.shape[1], curr_dog.shape[0]))

        # For each pixel (excluding border)
        for y in range(1, curr_dog.shape[0] - 1):
            for x in range(1, curr_dog.shape[1] - 1):
                # Get 3x3x3 neighborhood
                neighborhood = []

                # Previous DoG
                neighborhood.extend(prev_dog_resized[y - 1 : y + 2, x - 1 : x + 2].flatten())

                # Current DoG (excluding center)
                curr_neighbors = curr_dog[y - 1 : y + 2, x - 1 : x + 2].flatten()
                center_value = curr_dog[y, x]
                neighborhood.extend(curr_neighbors[:4])  # First 4 neighbors
                neighborhood.extend(curr_neighbors[5:])  # Last 4 neighbors (skip center)

                # Next DoG
                neighborhood.extend(next_dog_resized[y - 1 : y + 2, x - 1 : x + 2].flatten())

                # Convert to numpy array
                neighborhood = np.array(neighborhood)

                # Check if center is maximum
                if center_value > threshold and center_value > np.max(neighborhood):
                    extrema.append((octave, y, x, True))

                # Check if center is minimum
                elif center_value < -threshold and center_value < np.min(neighborhood):
                    extrema.append((octave, y, x, False))

    return extrema


def compute_orientation(image: np.ndarray, x: float, y: float, scale: float, num_bins: int = 36) -> float:
    """
    Compute dominant orientation for feature point

    Args:
        image: Input image
        x, y: Feature coordinates
        scale: Feature scale
        num_bins: Number of orientation histogram bins

    Returns:
        Dominant orientation in radians
    """
    # Determine region size based on scale
    region_size = max(int(6 * scale), 3)

    # Extract region
    x_int, y_int = int(x), int(y)
    x_min = max(0, x_int - region_size)
    x_max = min(image.shape[1] - 1, x_int + region_size)
    y_min = max(0, y_int - region_size)
    y_max = min(image.shape[0] - 1, y_int + region_size)

    # Initialize histogram
    hist = np.zeros(num_bins, dtype=np.float32)

    # Compute gradients and build histogram
    for j in range(y_min, y_max):
        for i in range(x_min, x_max):
            # Skip border pixels
            if i == 0 or i == image.shape[1] - 1 or j == 0 or j == image.shape[0] - 1:
                continue

            # Compute gradient
            dx = image[j, i + 1] - image[j, i - 1]
            dy = image[j + 1, i] - image[j - 1, i]

            # Compute magnitude and angle
            magnitude = math.sqrt(dx * dx + dy * dy)
            angle = math.atan2(dy, dx)

            # Convert angle to degrees in [0, 360)
            angle_deg = (angle * 180 / math.pi + 360) % 360

            # Determine histogram bin
            bin_idx = int(angle_deg * num_bins / 360)

            # Weight by magnitude and distance from center
            weight = magnitude * math.exp(-((i - x) ** 2 + (j - y) ** 2) / (2 * scale**2))
            hist[bin_idx] += weight

    # Smooth histogram
    smoothed = np.zeros_like(hist)
    for i in range(num_bins):
        prev_idx = (i - 1) % num_bins
        next_idx = (i + 1) % num_bins
        smoothed[i] = 0.25 * hist[prev_idx] + 0.5 * hist[i] + 0.25 * hist[next_idx]

    # Find dominant orientation (maximum bin)
    max_bin = np.argmax(smoothed)

    # Convert bin to angle in radians
    angle = (max_bin * 2 * math.pi / num_bins) - math.pi

    return angle
