"""
Tests for MindAR detector module.
Tests feature detection algorithms and cross-version compatibility.
"""

import time

import numpy as np
import pytest

from mindar.detector import Detector, DetectorConfig, fast_harris_response
from mindar.types import FeaturePoint


class TestDetectorConfig:
    """Test DetectorConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DetectorConfig()

        assert config.method == "super_hybrid"
        assert config.max_features == 1000
        assert config.fast_threshold == 20
        assert config.edge_threshold == 4.0
        assert config.harris_k == 0.04
        assert config.debug_mode is False
        assert config.enable_threading is False
        assert config.enable_subpixel is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DetectorConfig(
            method="fast",
            max_features=500,
            fast_threshold=15,
            edge_threshold=3.0,
            harris_k=0.05,
            debug_mode=True,
            enable_threading=True,
            enable_subpixel=True,
        )

        assert config.method == "fast"
        assert config.max_features == 500
        assert config.fast_threshold == 15
        assert config.edge_threshold == 3.0
        assert config.harris_k == 0.05
        assert config.debug_mode is True
        assert config.enable_threading is True
        assert config.enable_subpixel is True


class TestDetector:
    """Test Detector class functionality."""

    def test_detector_initialization(self):
        """Test detector initialization."""
        config = DetectorConfig(method="fast", debug_mode=True)
        detector = Detector(config)

        assert detector.config == config
        assert hasattr(detector, "fast_detector")
        assert hasattr(detector, "orb_detector")
        assert len(detector.detection_times) == 0

    def test_detector_default_initialization(self):
        """Test detector with default config."""
        detector = Detector()

        assert detector.config.method == "super_hybrid"
        assert detector.config.max_features == 1000
        assert hasattr(detector, "fast_detectors")  # For super_hybrid method

    def test_prepare_image_grayscale(self):
        """Test image preparation for grayscale input."""
        detector = Detector()

        # Test with grayscale image
        gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        prepared = detector._prepare_image(gray_image)

        assert prepared.shape == (100, 100)
        assert prepared.dtype == np.uint8
        assert np.array_equal(prepared, gray_image)

    def test_prepare_image_color(self):
        """Test image preparation for color input."""
        detector = Detector()

        # Test with color image
        color_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        prepared = detector._prepare_image(color_image)

        assert prepared.shape == (100, 100)
        assert prepared.dtype == np.uint8
        assert len(prepared.shape) == 2  # Should be grayscale

    def test_prepare_image_float(self):
        """Test image preparation for float input."""
        detector = Detector()

        # Test with float image (0-1 range)
        float_image = np.random.random((100, 100)).astype(np.float32)
        prepared = detector._prepare_image(float_image)

        assert prepared.shape == (100, 100)
        assert prepared.dtype == np.uint8
        assert prepared.max() <= 255
        assert prepared.min() >= 0

    def test_detect_fast_method(self, sample_image):
        """Test FAST detection method."""
        config = DetectorConfig(method="fast", max_features=100, debug_mode=True)
        detector = Detector(config)

        result = detector.detect(sample_image)

        assert "feature_points" in result
        assert isinstance(result["feature_points"], list)
        assert len(result["feature_points"]) > 0

        # Check feature point properties
        for point in result["feature_points"]:
            assert isinstance(point, FeaturePoint)
            assert hasattr(point, "x")
            assert hasattr(point, "y")
            assert hasattr(point, "descriptors")
            assert hasattr(point, "maxima")
            assert hasattr(point, "response")
            assert hasattr(point, "quality")

    def test_detect_harris_method(self, sample_image):
        """Test Harris detection method."""
        config = DetectorConfig(method="harris", max_features=100, debug_mode=True)
        detector = Detector(config)

        result = detector.detect(sample_image)

        assert "feature_points" in result
        assert isinstance(result["feature_points"], list)
        assert len(result["feature_points"]) > 0

    def test_detect_orb_method(self, sample_image):
        """Test ORB detection method."""
        config = DetectorConfig(method="orb", max_features=100, debug_mode=True)
        detector = Detector(config)

        result = detector.detect(sample_image)

        assert "feature_points" in result
        assert isinstance(result["feature_points"], list)
        assert len(result["feature_points"]) > 0

    def test_detect_super_hybrid_method(self, sample_image):
        """Test super hybrid detection method."""
        config = DetectorConfig(method="super_hybrid", max_features=100, debug_mode=True)
        detector = Detector(config)

        result = detector.detect(sample_image)

        assert "feature_points" in result
        assert isinstance(result["feature_points"], list)
        assert len(result["feature_points"]) > 0

    def test_detect_empty_image(self):
        """Test detection on empty/black image."""
        detector = Detector()

        # Create empty image
        empty_image = np.zeros((100, 100), dtype=np.uint8)
        result = detector.detect(empty_image)

        assert "feature_points" in result
        assert isinstance(result["feature_points"], list)
        # Should have very few or no features on empty image
        assert len(result["feature_points"]) <= 10

    def test_detect_noisy_image(self):
        """Test detection on noisy image."""
        detector = Detector()

        # Create noisy image
        noisy_image = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        result = detector.detect(noisy_image)

        assert "feature_points" in result
        assert isinstance(result["feature_points"], list)
        # Should detect some features even in noise

    def test_feature_point_quality(self, sample_image):
        """Test feature point quality computation."""
        detector = Detector()

        # Test corner quality computation
        quality = detector._compute_corner_quality(sample_image, 50, 50, window_size=5)
        assert isinstance(quality, float)
        assert 0.0 <= quality <= 1.0

    def test_harris_response_computation(self, sample_image):
        """Test Harris response computation."""
        detector = Detector()

        # Test Harris response computation
        response = detector._compute_harris_response(sample_image, 50, 50, window_size=3)
        assert isinstance(response, float)
        assert response >= 0.0

    def test_performance_stats(self, sample_image):
        """Test performance statistics collection."""
        detector = Detector()

        # Run detection multiple times
        for _ in range(3):
            detector.detect(sample_image)

        stats = detector.get_performance_stats()

        assert "avg_time_ms" in stats
        assert "min_time_ms" in stats
        assert "max_time_ms" in stats
        assert "total_detections" in stats
        assert stats["total_detections"] == 3
        assert stats["avg_time_ms"] > 0.0


class TestFastHarrisResponse:
    """Test fast Harris response computation."""

    def test_fast_harris_response(self, sample_image):
        """Test fast Harris response function."""
        response = fast_harris_response(sample_image, 50, 50, window_size=3)

        assert isinstance(response, float)
        assert response >= 0.0

    def test_fast_harris_response_edge_cases(self):
        """Test fast Harris response with edge cases."""
        # Test with small image
        small_image = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        response = fast_harris_response(small_image, 5, 5, window_size=3)
        assert isinstance(response, float)

        # Test with boundary coordinates
        response = fast_harris_response(small_image, 0, 0, window_size=3)
        assert isinstance(response, float)


class TestDetectorCompatibility:
    """Test detector compatibility across different scenarios."""

    def test_different_image_sizes(self):
        """Test detection with different image sizes."""
        detector = Detector()

        sizes = [(50, 50), (100, 100), (200, 200), (400, 400)]

        for width, height in sizes:
            image = np.random.randint(0, 255, (height, width), dtype=np.uint8)
            result = detector.detect(image)

            assert "feature_points" in result
            assert isinstance(result["feature_points"], list)

    def test_different_image_types(self):
        """Test detection with different image data types."""
        detector = Detector()

        # Test different dtypes
        dtypes = [np.uint8, np.uint16, np.float32, np.float64]

        for dtype in dtypes:
            if dtype in [np.uint8, np.uint16]:
                image = np.random.randint(0, 255, (100, 100), dtype=dtype)
            else:
                image = np.random.random((100, 100)).astype(dtype)

            result = detector.detect(image)
            assert "feature_points" in result

    def test_detection_methods_comparison(self, sample_image):
        """Compare different detection methods."""
        methods = ["fast", "harris", "orb", "super_hybrid"]
        results = {}

        for method in methods:
            config = DetectorConfig(method=method, max_features=100)
            detector = Detector(config)

            start_time = time.time()
            result = detector.detect(sample_image)
            detection_time = time.time() - start_time

            results[method] = {"feature_count": len(result["feature_points"]), "detection_time": detection_time}

        # All methods should produce some features
        for method, data in results.items():
            assert data["feature_count"] > 0
            assert data["detection_time"] > 0.0

    def test_feature_filtering(self, sample_image):
        """Test feature filtering and sorting."""
        detector = Detector()

        # Get raw features
        result = detector.detect(sample_image)
        features = result["feature_points"]

        if len(features) > 1:
            # Check that features are sorted by quality (descending)
            qualities = [f.quality for f in features]
            assert qualities == sorted(qualities, reverse=True)

            # Check that all features have valid quality scores
            for feature in features:
                assert 0.0 <= feature.quality <= 1.0
                assert feature.response > 0.0

    def test_descriptor_generation(self, sample_image):
        """Test descriptor generation for features."""
        detector = Detector()

        result = detector.detect(sample_image)
        features = result["feature_points"]

        for feature in features:
            assert hasattr(feature, "descriptors")
            assert isinstance(feature.descriptors, list)
            assert len(feature.descriptors) > 0

            # Check descriptor values are integers
            for desc in feature.descriptors:
                assert isinstance(desc, int)
                assert 0 <= desc <= 255


@pytest.mark.performance
class TestDetectorPerformance:
    """Performance tests for detector."""

    def test_detection_performance(self, sample_image):
        """Test detection performance."""
        detector = Detector()

        # Warm up
        detector.detect(sample_image)

        # Performance test
        start_time = time.time()
        for _ in range(10):
            detector.detect(sample_image)
        total_time = time.time() - start_time

        avg_time = total_time / 10
        assert avg_time < 1.0  # Should be under 1 second per detection

    def test_memory_usage(self, sample_image):
        """Test memory usage during detection."""
        import os

        import psutil

        detector = Detector()

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Run detection multiple times
        for _ in range(5):
            detector.detect(sample_image)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024
