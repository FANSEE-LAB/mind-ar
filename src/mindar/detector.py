"""
High-performance feature detector for ARtify.

This module provides optimized feature detection for real-time AR applications,
focusing on edge device performance with minimal dependencies.
"""

import time
from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np
from numba import jit

from .types import FeaturePoint

NUMBA_AVAILABLE = True

# Constants optimized for real-time performance
DEFAULT_MAX_FEATURES = 1000  # Increased for better matching
DEFAULT_FAST_THRESHOLD = 20  # Higher threshold for more stable features
DEFAULT_EDGE_THRESHOLD = 4.0
DEFAULT_HARRIS_K = 0.04
MIN_RESPONSE_THRESHOLD = 0.01  # Higher threshold for quality features

# Performance optimization flags
ENABLE_FAST_DETECTION = True
ENABLE_HARRIS_FILTERING = True
ENABLE_NON_MAX_SUPPRESSION = True
ENABLE_SUBPIXEL_REFINEMENT = False

# New quality thresholds
MIN_CORNER_QUALITY = 0.1
CORNER_QUALITY_LEVEL = 0.01
MIN_DISTANCE_BETWEEN_CORNERS = 10.0

# Super hybrid detection parameters
SUPER_HYBRID_MIN_QUALITY = 0.2
SUPER_HYBRID_SPATIAL_FILTER = True
SUPER_HYBRID_ADAPTIVE_THRESHOLD = True


@dataclass
class DetectorConfig:
    """Configuration for the detector to reduce parameter count."""

    method: str = "super_hybrid"
    max_features: int = DEFAULT_MAX_FEATURES
    fast_threshold: int = DEFAULT_FAST_THRESHOLD
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD
    harris_k: float = DEFAULT_HARRIS_K
    debug_mode: bool = False
    enable_threading: bool = False  # Add missing attribute
    enable_subpixel: bool = False  # Add missing attribute


class Detector:
    """
    High-performance feature detector optimized for ARtify applications.

    This detector supports multiple detection methods and is optimized for
    real-time performance on edge devices like Raspberry Pi.
    """

    def __init__(self, config: DetectorConfig = None):
        """
        Initialize the feature detector.

        Args:
            config: Detector configuration object
        """
        if config is None:
            config = DetectorConfig()

        self.config = config

        # Performance tracking
        self.detection_times = []

        # Initialize detectors
        self._init_detectors()

    def _init_detectors(self):
        """Initialize OpenCV detectors."""
        # FAST detector with optimized settings
        self.fast_detector = cv2.FastFeatureDetector_create(
            threshold=self.config.fast_threshold,
            nonmaxSuppression=ENABLE_NON_MAX_SUPPRESSION,
            type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16,
        )

        # ORB detector for descriptors
        self.orb_detector = cv2.ORB_create(
            nfeatures=self.config.max_features,
            scaleFactor=1.15,
            nlevels=6,
            edgeThreshold=10,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_FAST_SCORE,
            patchSize=25,
            fastThreshold=5,
        )

        # Good features to track parameters
        self.gftt_params = {
            "maxCorners": self.config.max_features,
            "qualityLevel": CORNER_QUALITY_LEVEL,
            "minDistance": MIN_DISTANCE_BETWEEN_CORNERS,
            "blockSize": 3,
            "useHarrisDetector": True,
            "k": self.config.harris_k,
        }

        # Harris corner detection parameters
        self.harris_block_size = 2
        self.harris_ksize = 3

        # Super hybrid detection settings
        if self.config.method == "super_hybrid":
            self._init_super_hybrid_detectors()

    def _init_super_hybrid_detectors(self):
        """Initialize detectors for super hybrid method."""
        # Use multiple FAST thresholds for robust detection
        self.fast_detectors = {
            "high": cv2.FastFeatureDetector_create(threshold=30, nonmaxSuppression=True),
            "medium": cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True),
            "low": cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True),
        }

        # Enhanced ORB for better descriptors
        self.enhanced_orb = cv2.ORB_create(
            nfeatures=self.config.max_features * 2,
            scaleFactor=1.15,
            nlevels=10,
            edgeThreshold=25,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=25,
        )

    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect features in the input image.

        Args:
            image: Input grayscale image (uint8)

        Returns:
            Dictionary with 'feature_points' list
        """
        start_time = time.time()

        # Ensure grayscale uint8
        gray = self._prepare_image(image)

        # Apply image enhancement for better feature detection
        gray = self._enhance_image(gray)

        # Detect features based on method
        feature_points = self._detect_by_method(gray)

        # Quality filtering and sorting
        feature_points = self._filter_and_sort_features(feature_points)

        # Add FREAK descriptors for MindAR compatibility
        self._add_freak_descriptors(gray, feature_points)

        # Track performance
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)

        if self.config.debug_mode:
            avg_quality = np.mean([fp.quality for fp in feature_points]) if feature_points else 0
            print(
                f"Detected {len(feature_points)} quality features in {detection_time*1000:.1f}ms (avg quality: {avg_quality:.3f})"
            )

        return {"feature_points": feature_points}

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Prepare image for feature detection."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)

        return gray

    def _detect_by_method(self, gray: np.ndarray) -> List[FeaturePoint]:
        """Detect features based on configured method."""
        if self.config.method == "fast":
            return self._detect_fast(gray)
        elif self.config.method == "harris":
            return self._detect_harris(gray)
        elif self.config.method == "orb":
            return self._detect_orb(gray)
        elif self.config.method == "hybrid":
            return self._detect_hybrid_enhanced(gray)
        elif self.config.method == "super_hybrid":
            return self._detect_super_hybrid(gray)
        else:
            raise ValueError(f"Unknown detection method: {self.config.method}")

    def _detect_super_hybrid(self, gray: np.ndarray) -> List[FeaturePoint]:
        """
        Super hybrid detection combining multiple methods with advanced filtering.
        """
        # Step 1: Multi-threshold FAST detection for robustness
        fast_features = self._detect_multi_threshold_fast(gray)
        fast_features = self._remove_spatial_duplicates(fast_features, threshold=8.0)

        # Step 2: Enhanced ORB detection
        orb_features = self._detect_enhanced_orb(gray)

        # Step 3: High-quality Harris corners for stable points
        harris_features = self._detect_harris(gray)

        # Step 4: Intelligent feature combination
        all_features = self._combine_features(orb_features, fast_features, harris_features)

        # Step 5: Spatial distribution optimization
        if SUPER_HYBRID_SPATIAL_FILTER:
            all_features = self._optimize_spatial_distribution(all_features, gray.shape)

        return all_features

    def _detect_multi_threshold_fast(self, gray: np.ndarray) -> List[FeaturePoint]:
        """Detect FAST features with multiple thresholds."""
        fast_features = []
        for name, detector in self.fast_detectors.items():
            keypoints = detector.detect(gray, None)
            features = self._convert_keypoints_with_quality(keypoints, gray)

            # Add threshold level as quality modifier
            threshold_bonus = {"high": 0.3, "medium": 0.2, "low": 0.1}[name]
            for feature_point in features:
                feature_point.quality += threshold_bonus
                fast_features.append(feature_point)

        return fast_features

    def _detect_enhanced_orb(self, gray: np.ndarray) -> List[FeaturePoint]:
        """Detect ORB features with enhanced quality calculation."""
        orb_keypoints, orb_descriptors = self.enhanced_orb.detectAndCompute(gray, None)
        orb_features = []

        if orb_keypoints and orb_descriptors is not None:
            for i, keypoint in enumerate(orb_keypoints):
                # Convert ORB descriptors
                desc_ints = self._convert_orb_descriptor(orb_descriptors, i)

                # Enhanced quality calculation for ORB
                quality = self._compute_corner_quality(gray, int(keypoint.pt[0]), int(keypoint.pt[1]))
                response_quality = min(keypoint.response / 100.0, 1.0)
                combined_quality = (quality + response_quality) / 2.0

                orb_features.append(
                    FeaturePoint(
                        x=float(keypoint.pt[0]),
                        y=float(keypoint.pt[1]),
                        scale=float(keypoint.size),
                        angle=float(keypoint.angle),
                        maxima=True,
                        response=float(keypoint.response),
                        descriptors=desc_ints,
                        quality=combined_quality,
                    )
                )

        return orb_features

    def _convert_orb_descriptor(self, orb_descriptors, index):
        """Convert ORB descriptor to integer list."""
        desc_ints = []
        if index < len(orb_descriptors):
            desc_bytes = orb_descriptors[index]
            for j in range(0, len(desc_bytes), 4):
                chunk = desc_bytes[j : j + 4]
                if len(chunk) == 4:
                    desc_ints.append(int.from_bytes(chunk, "big"))
        return desc_ints

    def _combine_features(self, orb_features, fast_features, harris_features) -> List[FeaturePoint]:
        """Combine features from different detectors intelligently."""
        all_features = []

        # Priority 1: High-quality ORB features
        for feature in orb_features:
            if feature.quality > SUPER_HYBRID_MIN_QUALITY:
                all_features.append(feature)

        # Priority 2: High-quality FAST features
        for feature in fast_features:
            if feature.quality > SUPER_HYBRID_MIN_QUALITY and not self._is_duplicate(
                feature, all_features, threshold=12.0
            ):
                all_features.append(feature)

        # Priority 3: Harris features for stability
        for feature in harris_features:
            if feature.quality > SUPER_HYBRID_MIN_QUALITY * 0.8 and not self._is_duplicate(
                feature, all_features, threshold=15.0
            ):
                all_features.append(feature)

        return all_features

    def _optimize_spatial_distribution(self, features: List[FeaturePoint], image_shape: tuple) -> List[FeaturePoint]:
        """Optimize spatial distribution of features."""
        if not features:
            return features

        height, width = image_shape
        grid_size = 4
        cell_width = width // grid_size
        cell_height = height // grid_size

        # Group features by grid cell
        grid_features = {}
        for feature in features:
            grid_x = min(int(feature.x // cell_width), grid_size - 1)
            grid_y = min(int(feature.y // cell_height), grid_size - 1)
            grid_key = (grid_x, grid_y)

            if grid_key not in grid_features:
                grid_features[grid_key] = []
            grid_features[grid_key].append(feature)

        # Select best features from each cell
        optimized_features = []
        max_per_cell = max(1, self.config.max_features // (grid_size * grid_size))

        for cell_features in grid_features.values():
            cell_features.sort(key=lambda f: f.quality * f.response, reverse=True)
            optimized_features.extend(cell_features[:max_per_cell])

        # Add remaining high-quality features if needed
        if len(optimized_features) < self.config.max_features:
            remaining_features = [f for f in features if f not in optimized_features]
            remaining_features.sort(key=lambda f: f.quality * f.response, reverse=True)

            for feature in remaining_features:
                if len(optimized_features) >= self.config.max_features:
                    break
                if not self._is_duplicate(feature, optimized_features, threshold=10.0):
                    optimized_features.append(feature)

        return optimized_features

    def _remove_spatial_duplicates(self, features: List[FeaturePoint], threshold: float = 8.0) -> List[FeaturePoint]:
        """Remove spatially duplicate features."""
        if not features:
            return features

        features.sort(key=lambda f: f.quality * f.response, reverse=True)
        filtered_features = []

        for feature in features:
            is_duplicate = False
            for existing in filtered_features:
                distance = np.sqrt((feature.x - existing.x) ** 2 + (feature.y - existing.y) ** 2)
                if distance < threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered_features.append(feature)

        return filtered_features

    def _enhance_image(self, gray: np.ndarray) -> np.ndarray:
        """Apply image enhancement for better feature detection."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0.5)
        return enhanced

    def _detect_fast(self, gray: np.ndarray) -> List[FeaturePoint]:
        """Fast corner detection with quality scoring."""
        keypoints = self.fast_detector.detect(gray, None)
        return self._convert_keypoints_with_quality(keypoints, gray)

    def _detect_harris(self, gray: np.ndarray) -> List[FeaturePoint]:
        """Harris corner detection with custom processing."""
        corners = cv2.goodFeaturesToTrack(gray, mask=None, **self.gftt_params)

        feature_points = []
        if corners is not None:
            corners = np.int0(corners)
            for corner in corners:
                x_coord, y_coord = corner.ravel()

                response = self._compute_harris_response(gray, x_coord, y_coord)
                quality = min(response / 1000.0, 1.0)

                feature_points.append(
                    FeaturePoint(
                        x=float(x_coord),
                        y=float(y_coord),
                        scale=1.0,
                        angle=0.0,
                        maxima=True,
                        response=float(response),
                        descriptors=[],
                        quality=quality,
                    )
                )

        return feature_points

    def _detect_orb(self, gray: np.ndarray) -> List[FeaturePoint]:
        """ORB feature detection with descriptors."""
        keypoints, descriptors = self.orb_detector.detectAndCompute(gray, None)

        feature_points = []
        for i, keypoint in enumerate(keypoints):
            desc_ints = self._convert_orb_descriptor(descriptors, i)
            quality = min(keypoint.response / 100.0, 1.0)

            feature_points.append(
                FeaturePoint(
                    x=float(keypoint.pt[0]),
                    y=float(keypoint.pt[1]),
                    scale=float(keypoint.size),
                    angle=float(keypoint.angle),
                    maxima=True,
                    response=float(keypoint.response),
                    descriptors=desc_ints,
                    quality=quality,
                )
            )

        return feature_points

    def _detect_hybrid_enhanced(self, gray: np.ndarray) -> List[FeaturePoint]:
        """Enhanced hybrid detection combining multiple methods."""
        fast_points = self._detect_fast(gray)
        harris_points = self._detect_harris(gray)
        orb_points = self._detect_orb(gray)

        all_points = []

        # Add high-quality Harris points first
        for harris_point in harris_points:
            if harris_point.quality > MIN_CORNER_QUALITY:
                all_points.append(harris_point)

        # Add high-response ORB points
        for orb_point in orb_points:
            if orb_point.quality > MIN_CORNER_QUALITY and not self._is_duplicate(orb_point, all_points):
                all_points.append(orb_point)

        # Add remaining FAST points if needed
        if len(all_points) < self.config.max_features // 2:
            for fast_point in fast_points:
                if len(all_points) >= self.config.max_features:
                    break
                if fast_point.quality > MIN_CORNER_QUALITY * 0.5 and not self._is_duplicate(fast_point, all_points):
                    all_points.append(fast_point)

        return all_points

    def _is_duplicate(self, point: FeaturePoint, existing_points: List[FeaturePoint], threshold: float = 10.0) -> bool:
        """Check if point is too close to existing points."""
        for existing_point in existing_points:
            distance = np.sqrt((point.x - existing_point.x) ** 2 + (point.y - existing_point.y) ** 2)
            if distance < threshold:
                return True
        return False

    def _filter_and_sort_features(self, feature_points: List[FeaturePoint]) -> List[FeaturePoint]:
        """Filter and sort features by quality and response."""
        filtered_points = [fp for fp in feature_points if fp.quality >= MIN_CORNER_QUALITY * 0.1]
        filtered_points.sort(key=lambda fp: fp.quality * fp.response, reverse=True)
        return filtered_points[: self.config.max_features]

    def _convert_keypoints_with_quality(self, keypoints, gray: np.ndarray) -> List[FeaturePoint]:
        """Convert OpenCV keypoints to FeaturePoint with quality scoring."""
        feature_points = []

        for keypoint in keypoints:
            x_coord, y_coord = int(keypoint.pt[0]), int(keypoint.pt[1])
            quality = self._compute_corner_quality(gray, x_coord, y_coord)

            feature_points.append(
                FeaturePoint(
                    x=float(keypoint.pt[0]),
                    y=float(keypoint.pt[1]),
                    scale=float(keypoint.size) if hasattr(keypoint, "size") else 1.0,
                    angle=float(keypoint.angle) if hasattr(keypoint, "angle") else 0.0,
                    maxima=True,
                    response=float(keypoint.response) if hasattr(keypoint, "response") else 1.0,
                    descriptors=[],
                    quality=quality,
                )
            )

        return feature_points

    def _compute_corner_quality(self, gray: np.ndarray, x_coord: int, y_coord: int, window_size: int = 5) -> float:
        """Compute corner quality based on local image statistics."""
        height, width = gray.shape
        half_window = window_size // 2

        if (
            x_coord - half_window < 0
            or x_coord + half_window >= width
            or y_coord - half_window < 0
            or y_coord + half_window >= height
        ):
            return 0.0

        window = gray[
            y_coord - half_window : y_coord + half_window + 1, x_coord - half_window : x_coord + half_window + 1
        ]

        contrast = np.std(window.astype(np.float32))
        grad_x = cv2.Sobel(window, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(window, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y).mean()

        quality = min((contrast * gradient_mag) / 10000.0, 1.0)
        return quality

    def _compute_harris_response(self, gray: np.ndarray, x_coord: int, y_coord: int, window_size: int = 3) -> float:
        """Compute Harris corner response at specific location."""
        height, width = gray.shape
        half_window = window_size // 2

        if (
            x_coord - half_window < 0
            or x_coord + half_window >= width
            or y_coord - half_window < 0
            or y_coord + half_window >= height
        ):
            return 0.0

        window = gray[
            y_coord - half_window : y_coord + half_window + 1, x_coord - half_window : x_coord + half_window + 1
        ]

        grad_x = cv2.Sobel(window, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(window, cv2.CV_32F, 0, 1, ksize=3)

        grad_xx = (grad_x * grad_x).sum()
        grad_yy = (grad_y * grad_y).sum()
        grad_xy = (grad_x * grad_y).sum()

        det = grad_xx * grad_yy - grad_xy * grad_xy
        trace = grad_xx + grad_yy

        if trace == 0:
            return 0.0

        return det - self.config.harris_k * (trace * trace)

    def _add_freak_descriptors(self, gray: np.ndarray, feature_points: List[FeaturePoint]):
        """Add FREAK descriptors to feature points for MindAR compatibility."""
        try:
            from .freak import FreakDescriptor

            if not feature_points:
                return

            freak = FreakDescriptor()

            keypoints = []
            for feature_point in feature_points:
                keypoints.append((feature_point.x, feature_point.y, feature_point.scale, feature_point.angle))

            descriptors = freak.compute_descriptors(gray, keypoints)

            for i, desc in enumerate(descriptors):
                if i < len(feature_points) and desc is not None:
                    if hasattr(desc, "tolist"):
                        feature_points[i].descriptors = desc.tolist()
                    else:
                        feature_points[i].descriptors = list(desc)

        except Exception as exception:
            if self.config.debug_mode:
                print(f"Warning: Failed to compute FREAK descriptors: {exception}")
            for feature_point in feature_points:
                if not feature_point.descriptors:
                    feature_point.descriptors = []

    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        if not self.detection_times:
            return {"avg_time_ms": 0, "fps": 0, "total_detections": 0}

        avg_time = np.mean(self.detection_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0

        return {
            "avg_time_ms": avg_time * 1000,
            "fps": fps,
            "total_detections": len(self.detection_times),
            "min_time_ms": min(self.detection_times) * 1000,
            "max_time_ms": max(self.detection_times) * 1000,
        }


@jit(nopython=True)
def fast_harris_response(image: np.ndarray, x_coord: int, y_coord: int, window_size: int = 3) -> float:
    """
    Fast Harris corner response computation using Numba JIT.
    Only used if Numba is available.
    """
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

    for delta_y in range(-window_size // 2, window_size // 2 + 1):
        for delta_x in range(-window_size // 2, window_size // 2 + 1):
            grad_x = (
                image[y_coord + delta_y, x_coord + delta_x + 1] - image[y_coord + delta_y, x_coord + delta_x - 1]
            ) * 0.5
            grad_y = (
                image[y_coord + delta_y + 1, x_coord + delta_x] - image[y_coord + delta_y - 1, x_coord + delta_x]
            ) * 0.5

            grad_xx += grad_x * grad_x
            grad_yy += grad_y * grad_y
            grad_xy += grad_x * grad_y

    det = grad_xx * grad_yy - grad_xy * grad_xy
    trace = grad_xx + grad_yy

    if trace == 0:
        return 0.0

    return det - DEFAULT_HARRIS_K * (trace * trace)
