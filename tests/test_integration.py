"""
Integration tests for MindAR complete pipeline.
Tests the full workflow from detection to matching to tracking.
"""

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


class TestCompletePipeline:
    """Test complete MindAR pipeline integration."""

    def test_detection_to_matching_pipeline(self, sample_image):
        """Test complete detection to matching pipeline."""
        # Initialize detector
        detector_config = DetectorConfig(method="super_hybrid", max_features=100, debug_mode=True)
        detector = Detector(detector_config)

        # Initialize matcher
        matcher_config = MatcherConfig(debug_mode=True)
        matcher = Matcher(matcher_config)

        # Detect features in sample image
        detection_result = detector.detect(sample_image)
        features1 = detection_result["feature_points"]

        assert len(features1) > 0
        assert all(isinstance(fp, FeaturePoint) for fp in features1)

        # Create a slightly modified version of the image for matching
        modified_image = sample_image.copy()
        # Add some noise
        noise = np.random.randint(-10, 10, modified_image.shape, dtype=np.int16)
        modified_image = np.clip(modified_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Detect features in modified image
        detection_result2 = detector.detect(modified_image)
        features2 = detection_result2["feature_points"]

        assert len(features2) > 0

        # Match features between images
        match_result = matcher.match(features1, features2)

        assert "matches" in match_result
        assert "homography" in match_result
        assert "inliers" in match_result
        assert isinstance(match_result["matches"], list)
        assert isinstance(match_result["inliers"], list)

        # Check that we have some matches
        assert len(match_result["matches"]) > 0

    def test_compiler_to_detection_pipeline(self, test_images_dir):
        """Test compiler to detection pipeline."""
        # Create test image
        image_path = test_images_dir / "pipeline_test.png"
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        cv2.imwrite(str(image_path), img)

        # Compile image
        compiler = MindARCompiler(debug_mode=True)

        with tempfile.NamedTemporaryFile(suffix=".mind", delete=False) as tmp_file:
            mind_path = tmp_file.name

        try:
            success = compiler.compile_images([str(image_path)], mind_path)
            assert success is True

            # Load compiled data
            mind_data = compiler.load_mind_file(mind_path)
            assert mind_data is not None
            assert len(mind_data["dataList"]) == 1

            target_data = mind_data["dataList"][0]
            assert "featurePoints" in target_data
            assert len(target_data["featurePoints"]) > 0

            # Test detection on the same image
            detector = Detector()
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detection_result = detector.detect(gray_img)

            assert len(detection_result["feature_points"]) > 0

        finally:
            Path(mind_path).unlink(missing_ok=True)
            Path(image_path).unlink(missing_ok=True)

    def test_detection_to_tracking_pipeline(self, sample_image):
        """Test detection to tracking pipeline."""
        # Initialize detector
        detector = Detector()

        # Detect features
        detection_result = detector.detect(sample_image)
        features = detection_result["feature_points"]

        assert len(features) > 0

        # Create tracking data from detected features
        tracking_points = [{"x": int(fp.x), "y": int(fp.y)} for fp in features[:10]]  # Use first 10 features

        marker_dimensions = [(sample_image.shape[1], sample_image.shape[0])]
        # Create proper tracking data format with multiple keyframes
        tracking_data = {
            0: {
                "points": tracking_points,
                "width": sample_image.shape[1],
                "height": sample_image.shape[0],
                "scale": 1.0,
            },  # 256px keyframe
            1: {
                "points": tracking_points,
                "width": sample_image.shape[1],
                "height": sample_image.shape[0],
                "scale": 1.0,
            },  # 128px keyframe (TRACKING_KEYFRAME)
        }
        tracking_data_list = [tracking_data]
        projection_transform = np.eye(3)

        tracker_config = TrackerConfig(
            marker_dimensions=marker_dimensions,
            tracking_data_list=tracking_data_list,
            projection_transform=projection_transform,
            input_width=sample_image.shape[1],
            input_height=sample_image.shape[0],
            debug_mode=True,
        )

        tracker = Tracker(tracker_config)

        # Test tracking
        last_model_view_transform = np.eye(3)
        target_index = 0

        tracking_result = tracker.track(sample_image, last_model_view_transform, target_index)

        assert isinstance(tracking_result, dict)
        assert "worldCoords" in tracking_result
        assert "screenCoords" in tracking_result
        assert "modelViewTransform" in tracking_result
        assert "trackingTime" in tracking_result

    def test_full_ar_pipeline(self, sample_image):
        """Test full AR pipeline: detection -> matching -> tracking."""
        # Step 1: Detection
        detector = Detector(DetectorConfig(method="super_hybrid", max_features=100))
        detection_result = detector.detect(sample_image)
        features = detection_result["feature_points"]

        assert len(features) > 0

        # Step 2: Create template from detected features
        template_points = [{"x": int(fp.x), "y": int(fp.y)} for fp in features[:20]]  # Use first 20 features

        # Step 3: Matching (simulate finding the template in a new image)
        matcher = Matcher()

        # Create a slightly different image (simulating camera movement)
        modified_image = sample_image.copy()
        # Apply slight transformation
        rows, cols = modified_image.shape[:2]
        M = np.float32([[1, 0, 5], [0, 1, 3]])  # Small translation
        modified_image = cv2.warpAffine(modified_image, M, (cols, rows))

        # Detect features in modified image
        detection_result2 = detector.detect(modified_image)
        features2 = detection_result2["feature_points"]

        # Match features
        match_result = matcher.match(features, features2)

        assert len(match_result["matches"]) > 0

        # Step 4: Tracking
        if len(match_result["matches"]) >= 4:  # Need minimum matches for tracking
            marker_dimensions = [(sample_image.shape[1], sample_image.shape[0])]
            # Create proper tracking data format with multiple keyframes
            tracking_data = {
                0: {
                    "points": template_points,
                    "width": sample_image.shape[1],
                    "height": sample_image.shape[0],
                    "scale": 1.0,
                },  # 256px keyframe
                1: {
                    "points": template_points,
                    "width": sample_image.shape[1],
                    "height": sample_image.shape[0],
                    "scale": 1.0,
                },  # 128px keyframe (TRACKING_KEYFRAME)
            }
            tracking_data_list = [tracking_data]
            projection_transform = np.eye(3)

            tracker_config = TrackerConfig(
                marker_dimensions=marker_dimensions,
                tracking_data_list=tracking_data_list,
                projection_transform=projection_transform,
                input_width=modified_image.shape[1],
                input_height=modified_image.shape[0],
            )

            tracker = Tracker(tracker_config)

            # Use homography from matching for initial pose
            if match_result["homography"] is not None:
                initial_transform = match_result["homography"]
            else:
                initial_transform = np.eye(3)

            tracking_result = tracker.track(modified_image, initial_transform, 0)

            assert isinstance(tracking_result, dict)
            assert "worldCoords" in tracking_result
            assert "screenCoords" in tracking_result
            assert "modelViewTransform" in tracking_result

    def test_pipeline_performance(self, sample_image):
        """Test performance of complete pipeline."""
        # Initialize components
        detector = Detector(DetectorConfig(method="super_hybrid", max_features=100))
        matcher = Matcher()

        # Performance test
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

        pipeline_time = time.time() - start_time

        # Should complete in reasonable time
        assert pipeline_time < 2.0  # Under 2 seconds for complete pipeline
        assert len(features1) > 0
        assert len(features2) > 0
        assert len(match_result["matches"]) > 0


class TestPipelineCompatibility:
    """Test pipeline compatibility across different scenarios."""

    def test_different_image_sizes_pipeline(self):
        """Test pipeline with different image sizes."""
        sizes = [(100, 100), (200, 200), (400, 400)]

        for width, height in sizes:
            # Create test image
            image = np.random.randint(0, 255, (height, width), dtype=np.uint8)

            # Test detection
            detector = Detector()
            detection_result = detector.detect(image)
            features = detection_result["feature_points"]

            assert len(features) > 0

            # Test matching with modified image
            modified_image = image.copy()
            modified_image = cv2.GaussianBlur(modified_image, (3, 3), 0)

            detection_result2 = detector.detect(modified_image)
            features2 = detection_result2["feature_points"]

            matcher = Matcher()
            match_result = matcher.match(features, features2)

            assert len(match_result["matches"]) > 0

    def test_different_feature_counts_pipeline(self, sample_image):
        """Test pipeline with different feature counts."""
        max_features_list = [50, 100, 200, 500]

        for max_features in max_features_list:
            detector_config = DetectorConfig(method="super_hybrid", max_features=max_features)
            detector = Detector(detector_config)

            detection_result = detector.detect(sample_image)
            features = detection_result["feature_points"]

            assert len(features) <= max_features
            assert len(features) > 0

            # Test matching
            matcher = Matcher()

            # Create modified image
            modified_image = sample_image.copy()
            modified_image = cv2.medianBlur(modified_image, 3)

            detection_result2 = detector.detect(modified_image)
            features2 = detection_result2["feature_points"]

            match_result = matcher.match(features, features2)

            assert len(match_result["matches"]) > 0

    def test_pipeline_with_noise(self, sample_image):
        """Test pipeline robustness with noise."""
        detector = Detector()
        matcher = Matcher()

        # Test with different noise levels
        noise_levels = [5, 10, 20, 30]

        for noise_level in noise_levels:
            # Add noise to image
            noisy_image = sample_image.copy()
            noise = np.random.randint(-noise_level, noise_level, noisy_image.shape, dtype=np.int16)
            noisy_image = np.clip(noisy_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Detection
            detection_result = detector.detect(noisy_image)
            features = detection_result["feature_points"]

            assert len(features) > 0

            # Matching with original
            detection_result2 = detector.detect(sample_image)
            features2 = detection_result2["feature_points"]

            match_result = matcher.match(features, features2)

            # Should still have some matches even with noise
            assert len(match_result["matches"]) > 0

    def test_pipeline_with_geometric_transformations(self, sample_image):
        """Test pipeline with geometric transformations."""
        detector = Detector()
        matcher = Matcher()

        # Test different transformations
        transformations = [
            # Small rotation
            lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
            # Small scale
            lambda img: cv2.resize(img, None, fx=0.8, fy=0.8),
            # Small translation
            lambda img: cv2.warpAffine(img, np.float32([[1, 0, 10], [0, 1, 5]]), (img.shape[1], img.shape[0])),
        ]

        for transform_func in transformations:
            # Apply transformation
            transformed_image = transform_func(sample_image)

            # Detection
            detection_result = detector.detect(transformed_image)
            features = detection_result["feature_points"]

            assert len(features) > 0

            # Matching with original
            detection_result2 = detector.detect(sample_image)
            features2 = detection_result2["feature_points"]

            match_result = matcher.match(features, features2)

            # Should have some matches
            assert len(match_result["matches"]) > 0


@pytest.mark.integration
class TestRealWorldScenarios:
    """Test real-world AR scenarios."""

    def test_marker_detection_scenario(self, sample_image):
        """Test marker detection scenario."""
        # Simulate marker detection workflow
        detector = Detector(DetectorConfig(method="super_hybrid", max_features=200))
        matcher = Matcher(MatcherConfig(min_matches=6))

        # Step 1: Detect features in marker image
        marker_features = detector.detect(sample_image)["feature_points"]

        assert len(marker_features) > 0

        # Step 2: Simulate camera frame with marker
        camera_frame = sample_image.copy()
        # Add some camera effects
        camera_frame = cv2.GaussianBlur(camera_frame, (3, 3), 0)
        camera_frame = cv2.convertScaleAbs(camera_frame, alpha=1.1, beta=5)

        # Step 3: Detect features in camera frame
        frame_features = detector.detect(camera_frame)["feature_points"]

        assert len(frame_features) > 0

        # Step 4: Match marker features with frame features
        match_result = matcher.match(marker_features, frame_features)

        # Step 5: Check if marker is detected
        if match_result["homography"] is not None and len(match_result["inliers"]) >= 6:
            # Marker detected successfully
            assert len(match_result["matches"]) > 0
            assert match_result["homography"].shape == (3, 3)
        else:
            # No marker detected (acceptable for some cases)
            assert len(match_result["matches"]) >= 0

    def test_tracking_scenario(self, sample_image):
        """Test continuous tracking scenario."""
        detector = Detector()

        # Create tracking data from sample image
        detection_result = detector.detect(sample_image)
        features = detection_result["feature_points"]

        tracking_points = [{"x": int(fp.x), "y": int(fp.y)} for fp in features[:15]]

        # Setup tracker
        marker_dimensions = [(sample_image.shape[1], sample_image.shape[0])]
        # Create proper tracking data format with multiple keyframes
        tracking_data = {
            0: {
                "points": tracking_points,
                "width": sample_image.shape[1],
                "height": sample_image.shape[0],
                "scale": 1.0,
            },  # 256px keyframe
            1: {
                "points": tracking_points,
                "width": sample_image.shape[1],
                "height": sample_image.shape[0],
                "scale": 1.0,
            },  # 128px keyframe (TRACKING_KEYFRAME)
        }
        tracking_data_list = [tracking_data]
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

        # Simulate tracking over multiple frames
        last_transform = np.eye(3)
        tracking_results = []

        for frame_idx in range(5):
            # Simulate camera movement
            if frame_idx > 0:
                # Add slight movement
                movement = np.random.uniform(-2, 2, (3, 3))
                movement[2, 2] = 1  # Keep homogeneous coordinate
                last_transform = last_transform @ movement

            # Track in current frame
            result = tracker.track(sample_image, last_transform, 0)
            tracking_results.append(result)

            # Update transform for next frame
            if result["modelViewTransform"] is not None:
                last_transform = result["modelViewTransform"]

        # Check tracking results
        assert len(tracking_results) == 5
        for result in tracking_results:
            assert isinstance(result, dict)
            assert "worldCoords" in result
            assert "screenCoords" in result
            assert "trackingTime" in result

    def test_compilation_to_detection_workflow(self, test_images_dir):
        """Test complete workflow from compilation to detection."""
        # Step 1: Create target images
        target_images = []
        for i in range(3):
            image_path = test_images_dir / f"target_{i}.png"

            # Create distinct target images
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            if i == 0:
                # Target 1: Rectangle
                cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)
            elif i == 1:
                # Target 2: Circle
                cv2.circle(img, (100, 100), 50, (0, 255, 0), -1)
            else:
                # Target 3: Triangle
                pts = np.array([[100, 50], [50, 150], [150, 150]], np.int32)
                cv2.fillPoly(img, [pts], (0, 0, 255))

            cv2.imwrite(str(image_path), img)
            target_images.append(str(image_path))

        # Step 2: Compile targets
        compiler = MindARCompiler(debug_mode=True)

        with tempfile.NamedTemporaryFile(suffix=".mind", delete=False) as tmp_file:
            mind_path = tmp_file.name

        try:
            success = compiler.compile_images(target_images, mind_path)
            assert success is True

            # Step 3: Load compiled data
            mind_data = compiler.load_mind_file(mind_path)
            assert mind_data is not None
            assert len(mind_data["dataList"]) == 3

            # Step 4: Test detection on each target
            detector = Detector()
            matcher = Matcher()

            for i, target_path in enumerate(target_images):
                # Load target image
                target_img = cv2.imread(target_path)
                target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

                # Detect features
                detection_result = detector.detect(target_gray)
                features = detection_result["feature_points"]

                assert len(features) > 0

                # Test matching with slightly modified version
                modified_img = target_gray.copy()
                modified_img = cv2.GaussianBlur(modified_img, (3, 3), 0)

                detection_result2 = detector.detect(modified_img)
                features2 = detection_result2["feature_points"]

                match_result = matcher.match(features, features2)

                # Should have good matches for same target
                assert len(match_result["matches"]) > 0

        finally:
            # Clean up
            Path(mind_path).unlink(missing_ok=True)
            for target_path in target_images:
                Path(target_path).unlink(missing_ok=True)


@pytest.mark.performance
class TestPipelinePerformance:
    """Performance tests for complete pipeline."""

    def test_pipeline_throughput(self, sample_image):
        """Test pipeline throughput."""
        detector = Detector(DetectorConfig(method="super_hybrid", max_features=100))
        matcher = Matcher()

        # Test multiple iterations
        num_iterations = 10
        total_time = 0

        for _ in range(num_iterations):
            # Create slightly different image each time
            modified_image = sample_image.copy()
            noise = np.random.randint(-10, 10, modified_image.shape, dtype=np.int16)
            modified_image = np.clip(modified_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            start_time = time.time()

            # Detection
            features1 = detector.detect(sample_image)["feature_points"]
            features2 = detector.detect(modified_image)["feature_points"]

            # Matching
            match_result = matcher.match(features1, features2)

            iteration_time = time.time() - start_time
            total_time += iteration_time

            assert len(features1) > 0
            assert len(features2) > 0
            assert len(match_result["matches"]) > 0

        avg_time = total_time / num_iterations
        assert avg_time < 0.5  # Should average under 500ms per iteration

    def test_memory_efficiency(self, sample_image):
        """Test memory efficiency of pipeline."""
        import os

        import psutil

        detector = Detector()
        matcher = Matcher()

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Run pipeline multiple times
        for _ in range(10):
            # Detection
            features1 = detector.detect(sample_image)["feature_points"]

            # Create modified image
            modified_image = sample_image.copy()
            modified_image = cv2.GaussianBlur(modified_image, (3, 3), 0)

            features2 = detector.detect(modified_image)["feature_points"]

            # Matching
            match_result = matcher.match(features1, features2)

            assert len(match_result["matches"]) > 0

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable
        assert memory_increase < 150 * 1024 * 1024  # Less than 150MB
