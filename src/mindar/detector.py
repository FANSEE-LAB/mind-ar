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

try:
    from numba import jit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        """Fallback decorator when numba is not available"""

        def decorator(func):
            return func

        return decorator


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
class FeaturePoint:
    """
    Optimized feature point representation.

    Attributes:
        x: X coordinate
        y: Y coordinate
        scale: Scale factor
        angle: Orientation angle in radians
        maxima: Whether this is a local maxima
        response: Corner response strength
        descriptors: List of descriptor values
        quality: Quality score (0-1)
    """

    x: float
    y: float
    scale: float
    angle: float
    maxima: bool
    response: float
    descriptors: List[int]
    quality: float = 0.0


class Detector:
    """
    High-performance feature detector optimized for ARtify applications.

    This detector supports multiple detection methods and is optimized for
    real-time performance on edge devices like Raspberry Pi.
    """

    def __init__(
        self,
        method: str = "super_hybrid",
        max_features: int = DEFAULT_MAX_FEATURES,
        fast_threshold: int = DEFAULT_FAST_THRESHOLD,
        edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
        harris_k: float = DEFAULT_HARRIS_K,
        debug_mode: bool = False,
    ):
        """
        Initialize the feature detector.

        Args:
            method: Detection method ('fast', 'harris', 'orb', 'hybrid', 'super_hybrid')
            max_features: Maximum number of features to detect
            fast_threshold: FAST corner detection threshold
            edge_threshold: Edge response threshold
            harris_k: Harris corner detector k parameter
            debug_mode: Enable debug output
        """
        self.method = method
        self.max_features = max_features
        self.fast_threshold = fast_threshold
        self.edge_threshold = edge_threshold
        self.harris_k = harris_k
        self.debug_mode = debug_mode

        # Performance tracking
        self.detection_times = []

        # Initialize detectors
        self._init_detectors()

    def _init_detectors(self):
        """Initialize OpenCV detectors."""
        # FAST detector with optimized settings
        self.fast_detector = cv2.FastFeatureDetector_create(
            threshold=self.fast_threshold,
            nonmaxSuppression=ENABLE_NON_MAX_SUPPRESSION,
            type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16,
        )

        # ORB detector for descriptors (High Sensitivity config from analysis)
        self.orb_detector = cv2.ORB_create(
            nfeatures=self.max_features,
            scaleFactor=1.15,  # Smaller scale factor for finer detection
            nlevels=6,  # Fewer levels for speed
            edgeThreshold=10,  # Lower edge threshold to detect more features
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_FAST_SCORE,  # Use FAST score for speed
            patchSize=25,  # Smaller patch size
            fastThreshold=5,  # Lower FAST threshold for more features
        )

        # Good features to track parameters
        self.gftt_params = dict(
            maxCorners=self.max_features,
            qualityLevel=CORNER_QUALITY_LEVEL,
            minDistance=MIN_DISTANCE_BETWEEN_CORNERS,
            blockSize=3,
            useHarrisDetector=True,
            k=self.harris_k,
        )

        # Harris corner detection parameters
        self.harris_block_size = 2
        self.harris_ksize = 3

        # Super hybrid detection settings
        if self.method == "super_hybrid":
            # Use multiple FAST thresholds for robust detection
            self.fast_detectors = {
                "high": cv2.FastFeatureDetector_create(threshold=30, nonmaxSuppression=True),
                "medium": cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True),
                "low": cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True),
            }

            # Enhanced ORB for better descriptors
            self.enhanced_orb = cv2.ORB_create(
                nfeatures=self.max_features * 2,  # More features for selection
                scaleFactor=1.15,  # Finer scale pyramid
                nlevels=10,  # More levels
                edgeThreshold=25,  # Lower edge threshold
                scoreType=cv2.ORB_HARRIS_SCORE,
                patchSize=25,  # Smaller patch for fine details
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
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)

        # Apply image enhancement for better feature detection
        gray = self._enhance_image(gray)

        # Detect features based on method
        if self.method == "fast":
            feature_points = self._detect_fast(gray)
        elif self.method == "harris":
            feature_points = self._detect_harris(gray)
        elif self.method == "orb":
            feature_points = self._detect_orb(gray)
        elif self.method == "hybrid":
            feature_points = self._detect_hybrid_enhanced(gray)
        elif self.method == "super_hybrid":
            feature_points = self._detect_super_hybrid(gray)
        else:
            raise ValueError(f"Unknown detection method: {self.method}")

        # Quality filtering and sorting
        feature_points = self._filter_and_sort_features(feature_points)

        # Add FREAK descriptors for MindAR compatibility
        self._add_freak_descriptors(gray, feature_points)

        # Track performance
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)

        if self.debug_mode:
            avg_quality = np.mean([fp.quality for fp in feature_points]) if feature_points else 0
            print(
                f"Detected {len(feature_points)} quality features in {detection_time*1000:.1f}ms (avg quality: {avg_quality:.3f})"
            )

        return {"feature_points": feature_points}

    def _detect_super_hybrid(self, gray: np.ndarray) -> List[FeaturePoint]:
        """
        Super hybrid detection combining multiple methods with advanced filtering.

        Based on test results:
        - FAST shows excellent discrimination (區分比 10.22)
        - ORB shows perfect separation (infinite discrimination)
        - Need to combine their strengths
        """
        all_features = []

        # Step 1: Multi-threshold FAST detection for robustness
        fast_features = []
        for name, detector in self.fast_detectors.items():
            keypoints = detector.detect(gray, None)
            features = self._convert_keypoints_with_quality(keypoints, gray)

            # Add threshold level as quality modifier
            threshold_bonus = {"high": 0.3, "medium": 0.2, "low": 0.1}[name]
            for fp in features:
                fp.quality += threshold_bonus
                fast_features.append(fp)

        # Remove duplicates from multi-threshold FAST
        fast_features = self._remove_spatial_duplicates(fast_features, threshold=8.0)

        # Step 2: Enhanced ORB detection
        orb_keypoints, orb_descriptors = self.enhanced_orb.detectAndCompute(gray, None)
        orb_features = []

        if orb_keypoints and orb_descriptors is not None:
            for i, kp in enumerate(orb_keypoints):
                # Convert ORB descriptors
                desc_ints = []
                if i < len(orb_descriptors):
                    desc_bytes = orb_descriptors[i]
                    for j in range(0, len(desc_bytes), 4):
                        chunk = desc_bytes[j : j + 4]
                        if len(chunk) == 4:
                            desc_ints.append(int.from_bytes(chunk, "big"))

                # Enhanced quality calculation for ORB
                quality = self._compute_corner_quality(gray, int(kp.pt[0]), int(kp.pt[1]))
                response_quality = min(kp.response / 100.0, 1.0)
                combined_quality = (quality + response_quality) / 2.0

                orb_features.append(
                    FeaturePoint(
                        x=float(kp.pt[0]),
                        y=float(kp.pt[1]),
                        scale=float(kp.size),
                        angle=float(kp.angle),
                        maxima=True,
                        response=float(kp.response),
                        descriptors=desc_ints,
                        quality=combined_quality,
                    )
                )

        # Step 3: High-quality Harris corners for stable points
        harris_features = self._detect_harris(gray)

        # Step 4: Intelligent feature combination

        # Priority 1: High-quality ORB features (perfect discrimination in tests)
        for feature in orb_features:
            if feature.quality > SUPER_HYBRID_MIN_QUALITY:
                all_features.append(feature)

        # Priority 2: High-quality FAST features (excellent discrimination)
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

        # Step 5: Spatial distribution optimization
        if SUPER_HYBRID_SPATIAL_FILTER:
            all_features = self._optimize_spatial_distribution(all_features, gray.shape)

        return all_features

    def _optimize_spatial_distribution(self, features: List[FeaturePoint], image_shape: tuple) -> List[FeaturePoint]:
        """優化特徵點的空間分佈，確保在圖像各區域都有代表性特徵"""
        if not features:
            return features

        height, width = image_shape

        # 將圖像分成網格
        grid_size = 4
        cell_width = width // grid_size
        cell_height = height // grid_size

        # 每個網格保留最好的特徵
        grid_features = {}

        for feature in features:
            grid_x = min(int(feature.x // cell_width), grid_size - 1)
            grid_y = min(int(feature.y // cell_height), grid_size - 1)
            grid_key = (grid_x, grid_y)

            if grid_key not in grid_features:
                grid_features[grid_key] = []
            grid_features[grid_key].append(feature)

        # 從每個網格選取最佳特徵
        optimized_features = []
        max_per_cell = max(1, self.max_features // (grid_size * grid_size))

        for cell_features in grid_features.values():
            # 按品質和響應排序
            cell_features.sort(key=lambda f: f.quality * f.response, reverse=True)
            optimized_features.extend(cell_features[:max_per_cell])

        # 如果特徵不足，添加其他高品質特徵
        if len(optimized_features) < self.max_features:
            remaining_features = [f for f in features if f not in optimized_features]
            remaining_features.sort(key=lambda f: f.quality * f.response, reverse=True)

            for feature in remaining_features:
                if len(optimized_features) >= self.max_features:
                    break
                if not self._is_duplicate(feature, optimized_features, threshold=10.0):
                    optimized_features.append(feature)

        return optimized_features

    def _remove_spatial_duplicates(self, features: List[FeaturePoint], threshold: float = 8.0) -> List[FeaturePoint]:
        """移除空間上重複的特徵點"""
        if not features:
            return features

        # 按品質排序，保留高品質的
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
        # Apply CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Slight Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0.5)

        return enhanced

    def _detect_fast(self, gray: np.ndarray) -> List[FeaturePoint]:
        """Fast corner detection with quality scoring."""
        keypoints = self.fast_detector.detect(gray, None)
        return self._convert_keypoints_with_quality(keypoints, gray)

    def _detect_harris(self, gray: np.ndarray) -> List[FeaturePoint]:
        """Harris corner detection with custom processing."""
        # Use goodFeaturesToTrack for better Harris corner detection
        corners = cv2.goodFeaturesToTrack(gray, mask=None, **self.gftt_params)

        feature_points = []
        if corners is not None:
            corners = np.int0(corners)
            for corner in corners:
                x, y = corner.ravel()

                # Compute Harris response for quality
                response = self._compute_harris_response(gray, x, y)
                quality = min(response / 1000.0, 1.0)  # Normalize to 0-1

                feature_points.append(
                    FeaturePoint(
                        x=float(x),
                        y=float(y),
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
        for i, kp in enumerate(keypoints):
            # Convert ORB descriptors to our format
            desc_ints = []
            if descriptors is not None and i < len(descriptors):
                desc_bytes = descriptors[i]
                # Pack 4 bytes into one 32-bit integer
                for j in range(0, len(desc_bytes), 4):
                    chunk = desc_bytes[j : j + 4]
                    if len(chunk) == 4:
                        desc_ints.append(int.from_bytes(chunk, "big"))

            # Quality based on response
            quality = min(kp.response / 100.0, 1.0)

            feature_points.append(
                FeaturePoint(
                    x=float(kp.pt[0]),
                    y=float(kp.pt[1]),
                    scale=float(kp.size),
                    angle=float(kp.angle),
                    maxima=True,
                    response=float(kp.response),
                    descriptors=desc_ints,
                    quality=quality,
                )
            )

        return feature_points

    def _detect_hybrid_enhanced(self, gray: np.ndarray) -> List[FeaturePoint]:
        """Enhanced hybrid detection combining multiple methods."""
        # Get FAST corners for speed
        fast_points = self._detect_fast(gray)

        # Get Harris corners for quality
        harris_points = self._detect_harris(gray)

        # Get ORB points for descriptors
        orb_points = self._detect_orb(gray)

        # Combine and deduplicate
        all_points = []

        # Add high-quality Harris points first
        for hp in harris_points:
            if hp.quality > MIN_CORNER_QUALITY:
                all_points.append(hp)

        # Add high-response ORB points
        for op in orb_points:
            if op.quality > MIN_CORNER_QUALITY and not self._is_duplicate(op, all_points):
                all_points.append(op)

        # Add remaining FAST points if needed
        if len(all_points) < self.max_features // 2:
            for fp in fast_points:
                if len(all_points) >= self.max_features:
                    break
                if fp.quality > MIN_CORNER_QUALITY * 0.5 and not self._is_duplicate(fp, all_points):
                    all_points.append(fp)

        return all_points

    def _is_duplicate(self, point: FeaturePoint, existing_points: List[FeaturePoint], threshold: float = 10.0) -> bool:
        """Check if point is too close to existing points."""
        for ep in existing_points:
            distance = np.sqrt((point.x - ep.x) ** 2 + (point.y - ep.y) ** 2)
            if distance < threshold:
                return True
        return False

    def _filter_and_sort_features(self, feature_points: List[FeaturePoint]) -> List[FeaturePoint]:
        """Filter and sort features by quality and response."""
        # Filter by minimum quality
        filtered_points = [fp for fp in feature_points if fp.quality >= MIN_CORNER_QUALITY * 0.1]

        # Sort by combined score (quality * response)
        filtered_points.sort(key=lambda fp: fp.quality * fp.response, reverse=True)

        # Take top features
        return filtered_points[: self.max_features]

    def _convert_keypoints_with_quality(self, keypoints, gray: np.ndarray) -> List[FeaturePoint]:
        """Convert OpenCV keypoints to FeaturePoint with quality scoring."""
        feature_points = []

        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])

            # Compute quality score based on local contrast
            quality = self._compute_corner_quality(gray, x, y)

            feature_points.append(
                FeaturePoint(
                    x=float(kp.pt[0]),
                    y=float(kp.pt[1]),
                    scale=float(kp.size) if hasattr(kp, "size") else 1.0,
                    angle=float(kp.angle) if hasattr(kp, "angle") else 0.0,
                    maxima=True,
                    response=float(kp.response) if hasattr(kp, "response") else 1.0,
                    descriptors=[],
                    quality=quality,
                )
            )

        return feature_points

    def _compute_corner_quality(self, gray: np.ndarray, x: int, y: int, window_size: int = 5) -> float:
        """Compute corner quality based on local image statistics."""
        h, w = gray.shape
        half_window = window_size // 2

        # Check bounds
        if x - half_window < 0 or x + half_window >= w or y - half_window < 0 or y + half_window >= h:
            return 0.0

        # Extract local window
        window = gray[y - half_window : y + half_window + 1, x - half_window : x + half_window + 1]

        # Compute local contrast (standard deviation)
        contrast = np.std(window.astype(np.float32))

        # Compute gradient magnitude
        gx = cv2.Sobel(window, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(window, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(gx * gx + gy * gy).mean()

        # Combine metrics (normalize to 0-1)
        quality = min((contrast * gradient_mag) / 10000.0, 1.0)

        return quality

    def _compute_harris_response(self, gray: np.ndarray, x: int, y: int, window_size: int = 3) -> float:
        """Compute Harris corner response at specific location."""
        h, w = gray.shape
        half_window = window_size // 2

        if x - half_window < 0 or x + half_window >= w or y - half_window < 0 or y + half_window >= h:
            return 0.0

        # Compute gradients in local window
        window = gray[y - half_window : y + half_window + 1, x - half_window : x + half_window + 1]

        # Sobel gradients
        Ix = cv2.Sobel(window, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(window, cv2.CV_32F, 0, 1, ksize=3)

        # Harris matrix elements
        Ixx = (Ix * Ix).sum()
        Iyy = (Iy * Iy).sum()
        Ixy = (Ix * Iy).sum()

        # Harris response
        det = Ixx * Iyy - Ixy * Ixy
        trace = Ixx + Iyy

        if trace == 0:
            return 0.0

        return det - self.harris_k * (trace * trace)

    def _add_freak_descriptors(self, gray: np.ndarray, feature_points: List[FeaturePoint]):
        """Add FREAK descriptors to feature points for MindAR compatibility."""
        try:
            from .freak import FreakDescriptor

            if not feature_points:
                return

            freak = FreakDescriptor()

            # Convert feature points to keypoint format
            keypoints = []
            for fp in feature_points:
                keypoints.append((fp.x, fp.y, fp.scale, fp.angle))

            # Compute FREAK descriptors
            descriptors = freak.compute_descriptors(gray, keypoints)

            # Add descriptors to feature points
            for i, desc in enumerate(descriptors):
                if i < len(feature_points) and desc is not None:
                    # Convert to list format
                    if hasattr(desc, "tolist"):
                        feature_points[i].descriptors = desc.tolist()
                    else:
                        feature_points[i].descriptors = list(desc)

        except Exception as e:
            if self.debug_mode:
                print(f"Warning: Failed to compute FREAK descriptors: {e}")
            # Fallback: add empty descriptors
            for fp in feature_points:
                if not fp.descriptors:
                    fp.descriptors = []

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
def fast_harris_response(image: np.ndarray, x: int, y: int, window_size: int = 3) -> float:
    """
    Fast Harris corner response computation using Numba JIT.
    Only used if Numba is available.
    """
    if x < window_size or x >= image.shape[1] - window_size or y < window_size or y >= image.shape[0] - window_size:
        return 0.0

    # Compute gradients
    Ixx = 0.0
    Iyy = 0.0
    Ixy = 0.0

    for dy in range(-window_size // 2, window_size // 2 + 1):
        for dx in range(-window_size // 2, window_size // 2 + 1):
            # Sobel-like gradients
            gx = (image[y + dy, x + dx + 1] - image[y + dy, x + dx - 1]) * 0.5
            gy = (image[y + dy + 1, x + dx] - image[y + dy - 1, x + dx]) * 0.5

            Ixx += gx * gx
            Iyy += gy * gy
            Ixy += gx * gy

    # Harris response
    det = Ixx * Iyy - Ixy * Ixy
    trace = Ixx + Iyy

    if trace == 0:
        return 0.0

    return det - DEFAULT_HARRIS_K * (trace * trace)
