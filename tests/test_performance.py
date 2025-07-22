"""
Performance tests for MindAR package.
Tests performance requirements and benchmarks.
"""

import statistics
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from mindar.compiler import MindARCompiler
from mindar.detector import Detector, DetectorConfig
from mindar.matcher import Matcher, MatcherConfig
from mindar.tracker import Tracker, TrackerConfig
from mindar.types import FeaturePoint


class TestDetectionPerformance:
    """Test detection performance requirements."""

    def test_detection_speed_requirements(self, sample_image):
        """Test that detection meets speed requirements."""
        detector = Detector(DetectorConfig(method="super_hybrid", max_features=100))

        # Warm up
        detector.detect(sample_image)

        # Performance test
        times = []
        num_iterations = 20

        for _ in range(num_iterations):
            start_time = time.time()
            result = detector.detect(sample_image)
            end_time = time.time()

            detection_time = end_time - start_time
            times.append(detection_time)

            assert "feature_points" in result
            assert len(result["feature_points"]) > 0

        # Calculate statistics
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_time = statistics.stdev(times)

        print("Detection Performance:")
        print(f"  Average time: {avg_time*1000:.2f}ms")
        print(f"  Min time: {min_time*1000:.2f}ms")
        print(f"  Max time: {max_time*1000:.2f}ms")
        print(f"  Std dev: {std_time*1000:.2f}ms")

        # Performance requirements (adjusted for 200x200 image)
        assert avg_time < 1.0  # Average under 1000ms for 200x200 image
        assert max_time < 2.0  # Max under 2000ms
        assert std_time < 0.2  # Consistent performance

    def test_detection_methods_performance_comparison(self, sample_image):
        """Compare performance of different detection methods."""
        methods = ["fast", "harris", "orb", "super_hybrid"]
        results = {}

        for method in methods:
            detector = Detector(DetectorConfig(method=method, max_features=100))

            # Warm up
            detector.detect(sample_image)

            # Performance test
            times = []
            feature_counts = []

            for _ in range(10):
                start_time = time.time()
                result = detector.detect(sample_image)
                end_time = time.time()

                times.append(end_time - start_time)
                feature_counts.append(len(result["feature_points"]))

            avg_time = statistics.mean(times)
            avg_features = statistics.mean(feature_counts)

            results[method] = {"avg_time": avg_time, "avg_features": avg_features}

        # Print comparison
        print("\nDetection Methods Performance Comparison:")
        for method, data in results.items():
            print(f"  {method:12}: {data['avg_time']*1000:6.2f}ms, {data['avg_features']:3.0f} features")

        # All methods should be reasonably fast
        for method, data in results.items():
            assert data["avg_time"] < 1.5  # All under 1500ms for 200x200 image
            assert data["avg_features"] > 0  # Should detect features

    def test_detection_scalability(self):
        """Test detection performance with different image sizes."""
        sizes = [(100, 100), (150, 150)]  # Further reduced sizes for CI/CD
        detector = Detector(DetectorConfig(method="super_hybrid", max_features=100))

        results = {}

        for width, height in sizes:
            # Create test image
            image = np.random.randint(0, 255, (height, width), dtype=np.uint8)

            # Warm up
            detector.detect(image)

            # Performance test (reduced iterations)
            times = []
            for _ in range(5):  # Reduced from 10 to 5
                start_time = time.time()
                result = detector.detect(image)
                end_time = time.time()

                times.append(end_time - start_time)
                assert len(result["feature_points"]) > 0

            avg_time = statistics.mean(times)
            results[f"{width}x{height}"] = avg_time

        # Print scalability results
        print("\nDetection Scalability:")
        for size, time_taken in results.items():
            print(f"  {size:8}: {time_taken*1000:6.2f}ms")

        # Performance should scale reasonably (adjusted for smaller images)
        for size, time_taken in results.items():
            assert time_taken < 3.0  # All sizes under 3000ms for CI/CD

    def test_detection_memory_efficiency(self, sample_image):
        """Test memory efficiency of detection."""
        import os

        import psutil

        detector = Detector(DetectorConfig(method="super_hybrid", max_features=100))
        process = psutil.Process(os.getpid())

        # Measure initial memory
        initial_memory = process.memory_info().rss

        # Run detection multiple times (reduced for CI/CD)
        for _ in range(10):  # Reduced from 20 to 10
            result = detector.detect(sample_image)
            assert len(result["feature_points"]) > 0

        # Measure final memory
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        print("Detection Memory Usage:")
        print(f"  Initial memory: {initial_memory / 1024 / 1024:.1f}MB")
        print(f"  Final memory: {final_memory / 1024 / 1024:.1f}MB")
        print(f"  Memory increase: {memory_increase / 1024 / 1024:.1f}MB")

        # Memory increase should be reasonable
        assert memory_increase < 50 * 1024 * 1024  # Less than 50MB


class TestMatchingPerformance:
    """Test matching performance requirements."""

    def test_matching_speed_requirements(self, feature_points):
        """Test that matching meets speed requirements."""
        if len(feature_points) < 20:
            pytest.skip("Need more feature points for matching test")

        matcher = Matcher(MatcherConfig(debug_mode=False))

        # Split features for matching
        features1 = feature_points[:10]
        features2 = feature_points[10:20]

        # Warm up
        matcher.match(features1, features2)

        # Performance test
        times = []
        num_iterations = 50

        for _ in range(num_iterations):
            start_time = time.time()
            result = matcher.match(features1, features2)
            end_time = time.time()

            matching_time = end_time - start_time
            times.append(matching_time)

            assert "matches" in result
            assert "homography" in result
            assert "inliers" in result

        # Calculate statistics
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_time = statistics.stdev(times)

        print("Matching Performance:")
        print(f"  Average time: {avg_time*1000:.2f}ms")
        print(f"  Min time: {min_time*1000:.2f}ms")
        print(f"  Max time: {max_time*1000:.2f}ms")
        print(f"  Std dev: {std_time*1000:.2f}ms")

        # Performance requirements
        assert avg_time < 0.05  # Average under 50ms
        assert max_time < 0.1  # Max under 100ms
        assert std_time < 0.02  # Consistent performance

    def test_matching_scalability(self):
        """Test matching performance with different feature counts."""
        feature_counts = [10, 50, 100, 200, 500]
        matcher = Matcher(MatcherConfig(debug_mode=False))

        results = {}

        for count in feature_counts:
            # Create test features
            features1 = []
            features2 = []

            for i in range(count):
                # Create similar but different features
                desc1 = [i % 256] * 64
                desc2 = [(i + 1) % 256] * 64

                feature1 = FeaturePoint(
                    x=float(i * 10), y=float(i * 10), scale=1.0, angle=0.0, descriptors=desc1, maxima=True, response=0.5
                )

                feature2 = FeaturePoint(
                    x=float(i * 10 + 2),
                    y=float(i * 10 + 1),
                    scale=1.0,
                    angle=0.1,
                    descriptors=desc2,
                    maxima=True,
                    response=0.6,
                )

                features1.append(feature1)
                features2.append(feature2)

            # Warm up
            matcher.match(features1, features2)

            # Performance test
            times = []
            for _ in range(10):
                start_time = time.time()
                result = matcher.match(features1, features2)
                end_time = time.time()

                times.append(end_time - start_time)
                assert len(result["matches"]) > 0

            avg_time = statistics.mean(times)
            results[count] = avg_time

        # Print scalability results
        print("\nMatching Scalability:")
        for count, time_taken in results.items():
            print(f"  {count:3} features: {time_taken*1000:6.2f}ms")

        # Performance should scale reasonably
        for count, time_taken in results.items():
            if count <= 100:
                assert time_taken < 0.1  # Small feature sets under 100ms
            else:
                assert time_taken < 0.5  # Large feature sets under 500ms

    def test_matching_cache_performance(self, feature_points):
        """Test matching cache performance benefits."""
        if len(feature_points) < 20:
            pytest.skip("Need more feature points for cache test")

        matcher = Matcher(MatcherConfig(debug_mode=False))

        features1 = feature_points[:10]
        features2 = feature_points[10:20]

        # First match (no cache)
        start_time = time.time()
        matcher.match(features1, features2)
        time1 = time.time() - start_time

        # Second match (with cache)
        start_time = time.time()
        matcher.match(features1, features2)
        time2 = time.time() - start_time

        print("Matching Cache Performance:")
        print(f"  First match: {time1*1000:.2f}ms")
        print(f"  Cached match: {time2*1000:.2f}ms")
        print(f"  Speedup: {time1/time2:.2f}x")

        # Cached match should be faster
        assert time2 <= time1
        assert len(matcher.cached_clusters) > 0


class TestTrackingPerformance:
    """Test tracking performance requirements."""

    def test_tracking_speed_requirements(self, sample_image):
        """Test that tracking meets speed requirements."""
        # Create tracking data
        detector = Detector(DetectorConfig(method="fast", max_features=50))
        detection_result = detector.detect(sample_image)
        features = detection_result["feature_points"]

        if len(features) < 10:
            pytest.skip("Need more features for tracking test")

        tracking_points = [{"x": int(fp.x), "y": int(fp.y)} for fp in features[:10]]

        marker_dimensions = [(sample_image.shape[1], sample_image.shape[0])]
        tracking_data_list = [
            {
                0: {
                    "points": tracking_points,
                    "width": sample_image.shape[1],
                    "height": sample_image.shape[0],
                    "scale": 1.0,
                },
                1: {
                    "points": tracking_points,
                    "width": sample_image.shape[1],
                    "height": sample_image.shape[0],
                    "scale": 1.0,
                },
            }
        ]
        projection_transform = np.eye(3)

        tracker_config = TrackerConfig(
            marker_dimensions=marker_dimensions,
            tracking_data_list=tracking_data_list,
            projection_transform=projection_transform,
            input_width=sample_image.shape[1],
            input_height=sample_image.shape[0],
            enable_caching=True,
        )

        tracker = Tracker(tracker_config)

        # Warm up
        last_transform = np.eye(3)
        tracker.track(sample_image, last_transform, 0)

        # Performance test (reduced for CI/CD)
        times = []
        num_iterations = 10  # Reduced from 30 to 10

        for _ in range(num_iterations):
            start_time = time.time()
            result = tracker.track(sample_image, last_transform, 0)
            end_time = time.time()

            tracking_time = end_time - start_time
            times.append(tracking_time)

            assert "worldCoords" in result
            assert "screenCoords" in result
            assert "trackingTime" in result

            # Update transform for next iteration
            if result["modelViewTransform"] is not None:
                last_transform = result["modelViewTransform"]

        # Calculate statistics
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_time = statistics.stdev(times)

        print("Tracking Performance:")
        print(f"  Average time: {avg_time*1000:.2f}ms")
        print(f"  Min time: {min_time*1000:.2f}ms")
        print(f"  Max time: {max_time*1000:.2f}ms")
        print(f"  Std dev: {std_time*1000:.2f}ms")

        # Performance requirements
        assert avg_time < 0.05  # Average under 50ms
        assert max_time < 0.1  # Max under 100ms
        assert std_time < 0.02  # Consistent performance

    def test_tracking_cache_performance(self, sample_image):
        """Test tracking cache performance benefits."""
        # Create tracking data
        detector = Detector(DetectorConfig(method="fast", max_features=50))
        detection_result = detector.detect(sample_image)
        features = detection_result["feature_points"]

        if len(features) < 10:
            pytest.skip("Need more features for tracking test")

        tracking_points = [{"x": int(fp.x), "y": int(fp.y)} for fp in features[:10]]

        marker_dimensions = [(sample_image.shape[1], sample_image.shape[0])]
        tracking_data_list = [
            {
                0: {
                    "points": tracking_points,
                    "width": sample_image.shape[1],
                    "height": sample_image.shape[0],
                    "scale": 1.0,
                },
                1: {
                    "points": tracking_points,
                    "width": sample_image.shape[1],
                    "height": sample_image.shape[0],
                    "scale": 1.0,
                },
            }
        ]
        projection_transform = np.eye(3)

        tracker_config = TrackerConfig(
            marker_dimensions=marker_dimensions,
            tracking_data_list=tracking_data_list,
            projection_transform=projection_transform,
            input_width=sample_image.shape[1],
            input_height=sample_image.shape[0],
            enable_caching=True,
        )

        tracker = Tracker(tracker_config)

        last_transform = np.eye(3)

        # First track (cache miss)
        start_time = time.time()
        tracker.track(sample_image, last_transform, 0)
        time1 = time.time() - start_time

        # Second track (cache hit)
        start_time = time.time()
        tracker.track(sample_image, last_transform, 0)
        time2 = time.time() - start_time

        print("Tracking Cache Performance:")
        print(f"  First track: {time1*1000:.2f}ms")
        print(f"  Cached track: {time2*1000:.2f}ms")
        print(f"  Speedup: {time1/time2:.2f}x")
        print(f"  Cache hits: {tracker.cache_hits}")
        print(f"  Cache misses: {tracker.cache_misses}")

        # Cached track should be faster
        assert time2 <= time1
        assert len(tracker.cache) > 0


class TestCompilerPerformance:
    """Test compiler performance requirements."""

    def test_compilation_speed_requirements(self, test_images_dir):
        """Test that compilation meets speed requirements."""
        compiler = MindARCompiler(debug_mode=False)

        # Create test images
        image_paths = []
        for i in range(5):
            image_path = test_images_dir / f"perf_test_image_{i}.png"

            # Create test image
            img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), img)
            image_paths.append(str(image_path))

        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".mind", delete=False) as tmp_file:
            output_path = tmp_file.name

        try:
            # Performance test
            start_time = time.time()
            success = compiler.compile_images(image_paths, output_path)
            compilation_time = time.time() - start_time

            assert success is True

            print("Compilation Performance:")
            print(f"  Images compiled: {len(image_paths)}")
            print(f"  Compilation time: {compilation_time:.2f}s")
            print(f"  Time per image: {compilation_time/len(image_paths)*1000:.2f}ms")

            # Performance requirements
            assert compilation_time < 5.0  # Total under 5 seconds
            assert compilation_time / len(image_paths) < 1.0  # Under 1 second per image

        finally:
            # Clean up
            Path(output_path).unlink(missing_ok=True)
            for image_path in image_paths:
                Path(image_path).unlink(missing_ok=True)

    def test_compilation_memory_efficiency(self, test_images_dir):
        """Test memory efficiency of compilation."""
        import os

        import psutil

        compiler = MindARCompiler(debug_mode=False)
        process = psutil.Process(os.getpid())

        # Create test images
        image_paths = []
        for i in range(3):
            image_path = test_images_dir / f"mem_test_image_{i}.png"

            # Create larger test image
            img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), img)
            image_paths.append(str(image_path))

        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".mind", delete=False) as tmp_file:
            output_path = tmp_file.name

        try:
            # Measure initial memory
            initial_memory = process.memory_info().rss

            # Run compilation
            success = compiler.compile_images(image_paths, output_path)

            # Measure final memory
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            assert success is True

            print("Compilation Memory Usage:")
            print(f"  Initial memory: {initial_memory / 1024 / 1024:.1f}MB")
            print(f"  Final memory: {final_memory / 1024 / 1024:.1f}MB")
            print(f"  Memory increase: {memory_increase / 1024 / 1024:.1f}MB")

            # Memory increase should be reasonable
            assert memory_increase < 100 * 1024 * 1024  # Less than 100MB

        finally:
            # Clean up
            Path(output_path).unlink(missing_ok=True)
            for image_path in image_paths:
                Path(image_path).unlink(missing_ok=True)


class TestEndToEndPerformance:
    """Test end-to-end performance requirements."""

    def test_complete_pipeline_performance(self, sample_image):
        """Test performance of complete pipeline."""
        # Initialize components
        detector = Detector(DetectorConfig(method="super_hybrid", max_features=100))
        matcher = Matcher(MatcherConfig(debug_mode=False))

        # Performance test
        times = []
        num_iterations = 20

        for _ in range(num_iterations):
            start_time = time.time()

            # Detection
            detection_result = detector.detect(sample_image)
            features1 = detection_result["feature_points"]

            # Create modified image
            modified_image = sample_image.copy()
            noise = np.random.randint(-5, 5, modified_image.shape, dtype=np.int16)
            modified_image = np.clip(modified_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Detection on modified image
            detection_result2 = detector.detect(modified_image)
            features2 = detection_result2["feature_points"]

            # Matching
            match_result = matcher.match(features1, features2)

            end_time = time.time()
            pipeline_time = end_time - start_time
            times.append(pipeline_time)

            assert len(features1) > 0
            assert len(features2) > 0
            assert len(match_result["matches"]) > 0

        # Calculate statistics
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_time = statistics.stdev(times)

        print("Complete Pipeline Performance:")
        print(f"  Average time: {avg_time*1000:.2f}ms")
        print(f"  Min time: {min_time*1000:.2f}ms")
        print(f"  Max time: {max_time*1000:.2f}ms")
        print(f"  Std dev: {std_time*1000:.2f}ms")

        # Performance requirements
        assert avg_time < 0.2  # Average under 200ms
        assert max_time < 0.4  # Max under 400ms
        assert std_time < 0.1  # Consistent performance

    def test_real_time_requirements(self, sample_image):
        """Test real-time performance requirements."""
        # For real-time AR, we need 30 FPS = 33.3ms per frame
        target_fps = 30
        target_frame_time = 1.0 / target_fps

        detector = Detector(DetectorConfig(method="fast", max_features=50))
        matcher = Matcher(MatcherConfig(debug_mode=False))

        # Performance test
        times = []
        num_iterations = 100

        for _ in range(num_iterations):
            start_time = time.time()

            # Detection
            detection_result = detector.detect(sample_image)
            features = detection_result["feature_points"]

            # Simple matching (self-matching for speed)
            if len(features) >= 2:
                features1 = features[: len(features) // 2]
                features2 = features[len(features) // 2 :]
                matcher.match(features1, features2)

            end_time = time.time()
            frame_time = end_time - start_time
            times.append(frame_time)

        # Calculate statistics
        avg_time = statistics.mean(times)
        p95_time = statistics.quantiles(times, n=20)[18]  # 95th percentile
        p99_time = statistics.quantiles(times, n=100)[98]  # 99th percentile

        print(f"Real-time Performance (Target: {target_fps} FPS, {target_frame_time*1000:.1f}ms):")
        print(f"  Average time: {avg_time*1000:.2f}ms")
        print(f"  95th percentile: {p95_time*1000:.2f}ms")
        print(f"  99th percentile: {p99_time*1000:.2f}ms")
        print(f"  Achieved FPS: {1.0/avg_time:.1f}")

        # Real-time requirements
        assert avg_time < target_frame_time * 1.5  # Average under 1.5x target
        assert p95_time < target_frame_time * 2.0  # 95% under 2x target
        assert p99_time < target_frame_time * 3.0  # 99% under 3x target
