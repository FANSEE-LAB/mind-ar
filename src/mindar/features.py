"""
Feature extraction and manipulation utilities.

This module contains utilities for feature extraction, manipulation, and visualization.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Feature:
    """
    A feature point with location, scale, and orientation.
    """

    x: float
    y: float
    scale: float
    angle: float
    response: float
    octave: int
    class_id: int = -1


def draw_features(image: np.ndarray, features: list, color=(0, 255, 0), radius=3, thickness=1) -> np.ndarray:
    """
    Draw feature points on an image.

    Args:
        image: Input image (BGR or grayscale)
        features: List of Feature objects or (x, y) tuples
        color: Color for drawing (BGR)
        radius: Circle radius
        thickness: Line thickness

    Returns:
        Image with features drawn
    """
    # Convert grayscale to BGR if needed
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()

    for feature in features:
        if hasattr(feature, "x") and hasattr(feature, "y"):
            # Feature object
            pt = (int(feature.x), int(feature.y))
        else:
            # Tuple
            pt = (int(feature[0]), int(feature[1]))

        cv2.circle(vis_image, pt, radius, color, thickness)

    return vis_image


def draw_matches(img1: np.ndarray, img2: np.ndarray, kp1: list, kp2: list, matches: list, flags: int = 0) -> np.ndarray:
    """
    Draw matches between two images.

    Args:
        img1: First image
        img2: Second image
        kp1: Keypoints from first image
        kp2: Keypoints from second image
        matches: List of match objects with queryIdx and trainIdx attributes
        flags: Drawing flags

    Returns:
        Image with matches drawn
    """
    # Convert keypoints to OpenCV format if needed
    if hasattr(kp1[0], "x") and not hasattr(kp1[0], "pt"):
        cv_kp1 = [cv2.KeyPoint(kp.x, kp.y, kp.scale) for kp in kp1]
    else:
        cv_kp1 = kp1

    if hasattr(kp2[0], "x") and not hasattr(kp2[0], "pt"):
        cv_kp2 = [cv2.KeyPoint(kp.x, kp.y, kp.scale) for kp in kp2]
    else:
        cv_kp2 = kp2

    # Draw matches
    return cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, matches, None, flags=flags)


def convert_to_cv_keypoints(features: list) -> list:
    """
    Convert features to OpenCV keypoints.

    Args:
        features: List of Feature objects

    Returns:
        List of cv2.KeyPoint objects
    """
    keypoints = []
    for feature in features:
        kp = cv2.KeyPoint()
        kp.pt = (feature.x, feature.y)
        kp.size = feature.scale
        kp.angle = feature.angle * 180 / np.pi  # Convert to degrees
        kp.response = feature.response if hasattr(feature, "response") else 0
        kp.octave = feature.octave if hasattr(feature, "octave") else 0
        kp.class_id = feature.class_id if hasattr(feature, "class_id") else -1
        keypoints.append(kp)

    return keypoints
