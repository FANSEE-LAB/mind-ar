"""
Tests for MindAR tracker module.
Tests tracking algorithms and cross-version compatibility.
"""

import time

import numpy as np
import pytest

from mindar.tracker import ScreenPoint, Tracker, TrackerConfig, TrackingPoint


class TestTrackingPoint:
    """Test TrackingPoint data structure."""

    def test_tracking_point_creation(self):
        """Test basic TrackingPoint creation."""
        point = TrackingPoint(x=100.0, y=200.0, z=300.0)

        assert point.x == 100.0
        assert point.y == 200.0
        assert point.z == 300.0


class TestScreenPoint:
    """Test ScreenPoint data structure."""

    def test_screen_point_creation(self):
        """Test basic ScreenPoint creation."""
        point = ScreenPoint(x=150.0, y=250.0)

        assert point.x == 150.0
        assert point.y == 250.0


class TestTrackerConfig:
    """Test TrackerConfig class."""

    def test_tracker_config_creation(self):
        """Test TrackerConfig creation."""
        marker_dimensions = [(100, 100), (200, 200)]
        tracking_data_list = [[{"points": [(50, 50), (150, 150)]}], [{"points": [(100, 100), (200, 200)]}]]
        projection_transform = np.eye(3)

        config = TrackerConfig(
            marker_dimensions=marker_dimensions,
            tracking_data_list=tracking_data_list,
            projection_transform=projection_transform,
            input_width=640,
            input_height=480,
            debug_mode=True,
            enable_threading=True,
            enable_caching=True,
        )

        assert config.marker_dimensions == marker_dimensions
        assert config.tracking_data_list == tracking_data_list
        assert np.array_equal(config.projection_transform, projection_transform)
        assert config.input_width == 640
        assert config.input_height == 480
        assert config.debug_mode is True
        assert config.enable_threading is True
        assert config.enable_caching is True


class TestTracker:
    """Test Tracker class functionality."""

    def test_tracker_initialization(self):
        """Test tracker initialization."""
        marker_dimensions = [(100, 100)]
        tracking_data_list = [[{"points": [(50, 50), (150, 150)]}]]
        projection_transform = np.eye(3)

        config = TrackerConfig(
            marker_dimensions=marker_dimensions,
            tracking_data_list=tracking_data_list,
            projection_transform=projection_transform,
            input_width=640,
            input_height=480,
            debug_mode=True,
        )

        tracker = Tracker(config)

        assert tracker.config == config
        assert hasattr(tracker, "tracking_keyframe_list")
        assert hasattr(tracker, "feature_points_list")
        assert hasattr(tracker, "image_pixels_list")
        assert hasattr(tracker, "image_properties_list")
        assert len(tracker.tracking_keyframe_list) == 1

    def test_tracker_prebuild(self):
        """Test tracker prebuild functionality."""
        marker_dimensions = [(100, 100)]
        tracking_data_list = [[{"points": [(50, 50), (150, 150)]}]]
        projection_transform = np.eye(3)

        config = TrackerConfig(
            marker_dimensions=marker_dimensions,
            tracking_data_list=tracking_data_list,
            projection_transform=projection_transform,
            input_width=640,
            input_height=480,
        )

        tracker = Tracker(config)

        # Test prebuild with sample tracking frame
        tracking_frame = {"points": [(50, 50), (150, 150)]}
        max_count = 10

        feature_points, image_pixels, image_properties = tracker._prebuild(tracking_frame, max_count)

        assert isinstance(feature_points, np.ndarray)
        assert isinstance(image_pixels, np.ndarray)
        assert isinstance(image_properties, np.ndarray)

    def test_model_view_projection_transform(self):
        """Test model-view-projection transform computation."""
        marker_dimensions = [(100, 100)]
        tracking_data_list = [[{"points": [(50, 50), (150, 150)]}]]
        projection_transform = np.eye(3)

        config = TrackerConfig(
            marker_dimensions=marker_dimensions,
            tracking_data_list=tracking_data_list,
            projection_transform=projection_transform,
            input_width=640,
            input_height=480,
        )

        tracker = Tracker(config)

        model_view_transform = np.eye(3)
        mvp_transform = tracker._build_model_view_projection_transform(projection_transform, model_view_transform)

        assert isinstance(mvp_transform, np.ndarray)
        assert mvp_transform.shape == (3, 3)

    def test_projection_computation(self, sample_image):
        """Test projection computation."""
        marker_dimensions = [(100, 100)]
        tracking_data_list = [[{"points": [(50, 50), (150, 150)]}]]
        projection_transform = np.eye(3)

        config = TrackerConfig(
            marker_dimensions=marker_dimensions,
            tracking_data_list=tracking_data_list,
            projection_transform=projection_transform,
            input_width=sample_image.shape[1],
            input_height=sample_image.shape[0],
        )

        tracker = Tracker(config)

        model_view_projection_transform = np.eye(3)
        target_index = 0

        projected_image = tracker._compute_projection(model_view_projection_transform, sample_image, target_index)

        assert isinstance(projected_image, np.ndarray)
        assert projected_image.shape == sample_image.shape

    def test_screen_coordinate_computation(self):
        """Test screen coordinate computation."""
        marker_dimensions = [(100, 100)]
        tracking_data_list = [[{"points": [(50, 50), (150, 150)]}]]
        projection_transform = np.eye(3)

        config = TrackerConfig(
            marker_dimensions=marker_dimensions,
            tracking_data_list=tracking_data_list,
            projection_transform=projection_transform,
            input_width=640,
            input_height=480,
        )

        tracker = Tracker(config)

        model_view_projection_transform = np.eye(3)
        world_x, world_y = 100.0, 200.0

        screen_point = tracker._compute_screen_coordinate(model_view_projection_transform, world_x, world_y)

        assert isinstance(screen_point, ScreenPoint)
        assert hasattr(screen_point, "x")
        assert hasattr(screen_point, "y")

    def test_normalized_correlation(self):
        """Test normalized correlation computation."""
        marker_dimensions = [(100, 100)]
        tracking_data_list = [[{"points": [(50, 50), (150, 150)]}]]
        projection_transform = np.eye(3)

        config = TrackerConfig(
            marker_dimensions=marker_dimensions,
            tracking_data_list=tracking_data_list,
            projection_transform=projection_transform,
            input_width=640,
            input_height=480,
        )

        tracker = Tracker(config)

        # Create test pixels
        target_pixels = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        template_pixels = np.random.randint(0, 255, (5, 5), dtype=np.uint8)

        correlation = tracker.compute_normalized_correlation(
            target_pixels=target_pixels,
            template_pixels=template_pixels,
            center_x=5,
            center_y=5,
            search_x=2,
            search_y=2,
            template_size=5,
        )

        assert isinstance(correlation, float)
        assert -1.0 <= correlation <= 1.0

    def test_tracking_basic(self, sample_image):
        """Test basic tracking functionality."""
        marker_dimensions = [(100, 100)]
        tracking_data_list = [[{"points": [(50, 50), (150, 150)]}]]
        projection_transform = np.eye(3)

        config = TrackerConfig(
            marker_dimensions=marker_dimensions,
            tracking_data_list=tracking_data_list,
            projection_transform=projection_transform,
            input_width=sample_image.shape[1],
            input_height=sample_image.shape[0],
        )

        tracker = Tracker(config)

        last_model_view_transform = np.eye(3)
        target_index = 0

        result = tracker.track(sample_image, last_model_view_transform, target_index)

        assert isinstance(result, dict)
        assert "worldCoords" in result
        assert "screenCoords" in result
        assert "modelViewTransform" in result
        assert "projectedImage" in result
        assert "trackingTime" in result

    def test_tracking_with_caching(self, sample_image):
        """Test tracking with caching enabled."""
        marker_dimensions = [(100, 100)]
        tracking_data_list = [[{"points": [(50, 50), (150, 150)]}]]
        projection_transform = np.eye(3)

        config = TrackerConfig(
            marker_dimensions=marker_dimensions,
            tracking_data_list=tracking_data_list,
            projection_transform=projection_transform,
            input_width=sample_image.shape[1],
            input_height=sample_image.shape[0],
            enable_caching=True,
        )

        tracker = Tracker(config)

        last_model_view_transform = np.eye(3)
        target_index = 0

        # First track (cache miss)
        result1 = tracker.track(sample_image, last_model_view_transform, target_index)

        # Second track (cache hit)
        result2 = tracker.track(sample_image, last_model_view_transform, target_index)

        assert tracker.cache_hits > 0
        assert tracker.cache_misses > 0
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)


class TestTrackerCompatibility:
    """Test tracker compatibility across different scenarios."""

    def test_different_image_sizes(self):
        """Test tracking with different image sizes."""
        sizes = [(320, 240), (640, 480), (1280, 720)]

        for width, height in sizes:
            marker_dimensions = [(100, 100)]
            tracking_data_list = [[{"points": [(50, 50), (150, 150)]}]]
            projection_transform = np.eye(3)

            config = TrackerConfig(
                marker_dimensions=marker_dimensions,
                tracking_data_list=tracking_data_list,
                projection_transform=projection_transform,
                input_width=width,
                input_height=height,
            )

            tracker = Tracker(config)

            # Create test image
            test_image = np.random.randint(0, 255, (height, width), dtype=np.uint8)
            last_model_view_transform = np.eye(3)
            target_index = 0

            result = tracker.track(test_image, last_model_view_transform, target_index)

            assert isinstance(result, dict)
            assert "worldCoords" in result
            assert "screenCoords" in result

    def test_different_marker_dimensions(self, sample_image):
        """Test tracking with different marker dimensions."""
        marker_dimensions_list = [(50, 50), (100, 100), (200, 200)]

        for marker_dim in marker_dimensions_list:
            marker_dimensions = [marker_dim]
            tracking_data_list = [[{"points": [(25, 25), (75, 75)]}]]
            projection_transform = np.eye(3)

            config = TrackerConfig(
                marker_dimensions=marker_dimensions,
                tracking_data_list=tracking_data_list,
                projection_transform=projection_transform,
                input_width=sample_image.shape[1],
                input_height=sample_image.shape[0],
            )

            tracker = Tracker(config)

            last_model_view_transform = np.eye(3)
            target_index = 0

            result = tracker.track(sample_image, last_model_view_transform, target_index)

            assert isinstance(result, dict)
            assert "worldCoords" in result
            assert "screenCoords" in result

    def test_multiple_targets(self, sample_image):
        """Test tracking with multiple targets."""
        marker_dimensions = [(100, 100), (150, 150)]
        tracking_data_list = [[{"points": [(50, 50), (150, 150)]}], [{"points": [(75, 75), (225, 225)]}]]
        projection_transform = np.eye(3)

        config = TrackerConfig(
            marker_dimensions=marker_dimensions,
            tracking_data_list=tracking_data_list,
            projection_transform=projection_transform,
            input_width=sample_image.shape[1],
            input_height=sample_image.shape[0],
        )

        tracker = Tracker(config)

        last_model_view_transform = np.eye(3)

        # Test tracking each target
        for target_index in range(len(marker_dimensions)):
            result = tracker.track(sample_image, last_model_view_transform, target_index)

            assert isinstance(result, dict)
            assert "worldCoords" in result
            assert "screenCoords" in result

    def test_different_transform_types(self, sample_image):
        """Test tracking with different transform types."""
        marker_dimensions = [(100, 100)]
        tracking_data_list = [[{"points": [(50, 50), (150, 150)]}]]

        # Test different projection transforms
        projection_transforms = [
            np.eye(3),
            np.array([[1.1, 0.1, 0], [0.1, 1.1, 0], [0, 0, 1]]),
            np.array([[0.9, -0.1, 10], [-0.1, 0.9, 20], [0, 0, 1]]),
        ]

        for projection_transform in projection_transforms:
            config = TrackerConfig(
                marker_dimensions=marker_dimensions,
                tracking_data_list=tracking_data_list,
                projection_transform=projection_transform,
                input_width=sample_image.shape[1],
                input_height=sample_image.shape[0],
            )

            tracker = Tracker(config)

            last_model_view_transform = np.eye(3)
            target_index = 0

            result = tracker.track(sample_image, last_model_view_transform, target_index)

            assert isinstance(result, dict)
            assert "worldCoords" in result
            assert "screenCoords" in result


@pytest.mark.performance
class TestTrackerPerformance:
    """Performance tests for tracker."""

    def test_tracking_performance(self, sample_image):
        """Test tracking performance."""
        marker_dimensions = [(100, 100)]
        tracking_data_list = [[{"points": [(50, 50), (150, 150)]}]]
        projection_transform = np.eye(3)

        config = TrackerConfig(
            marker_dimensions=marker_dimensions,
            tracking_data_list=tracking_data_list,
            projection_transform=projection_transform,
            input_width=sample_image.shape[1],
            input_height=sample_image.shape[0],
        )

        tracker = Tracker(config)

        last_model_view_transform = np.eye(3)
        target_index = 0

        # Warm up
        tracker.track(sample_image, last_model_view_transform, target_index)

        # Performance test
        start_time = time.time()
        for _ in range(10):
            tracker.track(sample_image, last_model_view_transform, target_index)
        total_time = time.time() - start_time

        avg_time = total_time / 10
        assert avg_time < 1.0  # Should be under 1 second per track

    def test_caching_performance(self, sample_image):
        """Test caching performance benefits."""
        marker_dimensions = [(100, 100)]
        tracking_data_list = [[{"points": [(50, 50), (150, 150)]}]]
        projection_transform = np.eye(3)

        config = TrackerConfig(
            marker_dimensions=marker_dimensions,
            tracking_data_list=tracking_data_list,
            projection_transform=projection_transform,
            input_width=sample_image.shape[1],
            input_height=sample_image.shape[0],
            enable_caching=True,
        )

        tracker = Tracker(config)

        last_model_view_transform = np.eye(3)
        target_index = 0

        # First track (no cache)
        start_time = time.time()
        tracker.track(sample_image, last_model_view_transform, target_index)
        time1 = time.time() - start_time

        # Second track (with cache)
        start_time = time.time()
        tracker.track(sample_image, last_model_view_transform, target_index)
        time2 = time.time() - start_time

        # Cached track should be faster
        assert time2 <= time1
        assert len(tracker.cache) > 0


class TestTrackerEdgeCases:
    """Test tracker edge cases and error handling."""

    def test_empty_tracking_data(self):
        """Test tracker with empty tracking data."""
        marker_dimensions = [(100, 100)]
        tracking_data_list = [[{"points": []}]]  # Empty points
        projection_transform = np.eye(3)

        config = TrackerConfig(
            marker_dimensions=marker_dimensions,
            tracking_data_list=tracking_data_list,
            projection_transform=projection_transform,
            input_width=640,
            input_height=480,
        )

        # Should handle empty tracking data gracefully
        tracker = Tracker(config)
        assert len(tracker.tracking_keyframe_list) == 1

    def test_single_point_tracking(self):
        """Test tracking with single point."""
        marker_dimensions = [(100, 100)]
        tracking_data_list = [[{"points": [(50, 50)]}]]  # Single point
        projection_transform = np.eye(3)

        config = TrackerConfig(
            marker_dimensions=marker_dimensions,
            tracking_data_list=tracking_data_list,
            projection_transform=projection_transform,
            input_width=640,
            input_height=480,
        )

        tracker = Tracker(config)

        test_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        last_model_view_transform = np.eye(3)
        target_index = 0

        result = tracker.track(test_image, last_model_view_transform, target_index)

        assert isinstance(result, dict)
        assert "worldCoords" in result
        assert "screenCoords" in result

    def test_extreme_coordinates(self):
        """Test tracking with extreme coordinate values."""
        marker_dimensions = [(100, 100)]
        tracking_data_list = [[{"points": [(1e6, 1e6), (-1e6, -1e6)]}]]
        projection_transform = np.eye(3)

        config = TrackerConfig(
            marker_dimensions=marker_dimensions,
            tracking_data_list=tracking_data_list,
            projection_transform=projection_transform,
            input_width=640,
            input_height=480,
        )

        tracker = Tracker(config)

        test_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        last_model_view_transform = np.eye(3)
        target_index = 0

        result = tracker.track(test_image, last_model_view_transform, target_index)

        assert isinstance(result, dict)
        assert "worldCoords" in result
        assert "screenCoords" in result
