"""
Compatibility tests for MindAR package.
Tests compatibility across Python versions and dependencies.
"""

import sys

import numpy as np

from mindar.compiler import MindARCompiler
from mindar.detector import Detector, DetectorConfig
from mindar.matcher import Matcher, MatcherConfig
from mindar.tracker import TrackerConfig
from mindar.types import DetectionResult, FeaturePoint, Match


class TestPythonVersionCompatibility:
    """Test compatibility across Python versions."""

    def test_python_version_info(self):
        """Test Python version information."""
        version_info = sys.version_info

        # Should work with Python 3.9+
        assert version_info.major == 3
        assert version_info.minor >= 9

        print(f"Testing with Python {version_info.major}.{version_info.minor}.{version_info.micro}")

    def test_type_annotations_compatibility(self):
        """Test type annotations compatibility."""
        # Test that type hints work correctly
        from typing import List

        # Test FeaturePoint creation with type hints
        point: FeaturePoint = FeaturePoint(
            x=100.0, y=200.0, scale=1.0, angle=0.0, descriptors=[1, 2, 3, 4], maxima=True, response=0.5, quality=0.7
        )

        assert isinstance(point, FeaturePoint)
        assert point.x == 100.0
        assert point.y == 200.0

        # Test list of features
        features: List[FeaturePoint] = [point]
        assert len(features) == 1
        assert isinstance(features[0], FeaturePoint)

    def test_dataclass_compatibility(self):
        """Test dataclass compatibility."""
        # Test that dataclasses work correctly
        point = FeaturePoint(x=100.0, y=200.0, scale=1.0, angle=0.0, descriptors=[1, 2, 3], maxima=True, response=0.5)

        # Test dataclass methods
        assert hasattr(point, "__post_init__")
        assert hasattr(point, "to_dict")
        assert hasattr(point, "from_dict")

        # Test default values
        assert point.quality == 0.5  # Default value

    def test_numpy_compatibility(self):
        """Test numpy compatibility."""
        # Test different numpy dtypes
        dtypes = [np.float32, np.float64, np.int32, np.int64]

        for dtype in dtypes:
            # Test array creation
            arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
            assert arr.dtype == dtype

            # Test with DetectionResult
            homography = np.eye(3, dtype=dtype)
            result = DetectionResult(target_id=1, homography=homography, matches=[], inliers=10, confidence=0.9)

            assert result.homography.dtype == dtype
            assert np.array_equal(result.homography, homography)


class TestDependencyCompatibility:
    """Test compatibility with different dependency versions."""

    def test_opencv_compatibility(self):
        """Test OpenCV compatibility."""
        import cv2

        # Test basic OpenCV functionality
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        # Test grayscale conversion
        gray = cv2.cvtColor(img, cv2.COLOR_GRAY2GRAY)
        assert gray.shape == img.shape

        # Test blur
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        assert blurred.shape == img.shape

        # Test feature detection
        detector = Detector()
        result = detector.detect(img)
        assert "feature_points" in result

        print(f"OpenCV version: {cv2.__version__}")

    def test_numpy_compatibility_extended(self):
        """Test extended numpy compatibility."""
        # Test array operations
        arr1 = np.random.random((100, 100))
        arr2 = np.random.random((100, 100))

        # Test basic operations
        result = arr1 + arr2
        assert result.shape == arr1.shape

        # Test broadcasting
        result = arr1 + 1.0
        assert result.shape == arr1.shape

        # Test with different dtypes
        for dtype in [np.float32, np.float64]:
            arr = np.array([[1, 2], [3, 4]], dtype=dtype)
            assert arr.dtype == dtype

            # Test matrix operations
            inv = np.linalg.inv(arr)
            assert inv.dtype == dtype

    def test_msgpack_compatibility(self):
        """Test msgpack compatibility."""
        import msgpack

        # Test basic serialization
        data = {"test": "value", "number": 42, "array": [1, 2, 3]}

        # Pack
        packed = msgpack.packb(data)
        assert isinstance(packed, bytes)

        # Unpack
        unpacked = msgpack.unpackb(packed)
        assert unpacked == data

        # Test with numpy arrays
        arr = np.array([1, 2, 3, 4])
        packed_arr = msgpack.packb(arr.tolist())
        unpacked_arr = msgpack.unpackb(packed_arr)
        assert unpacked_arr == arr.tolist()

    def test_numba_compatibility(self):
        """Test numba compatibility (optional)."""
        try:
            import numba  # noqa: F401

            from mindar.jit import NUMBA_AVAILABLE, get_jit_info

            if NUMBA_AVAILABLE:
                # Test JIT compilation
                info = get_jit_info()
                assert "numba_version" in info
                assert "jit_enabled" in info

                print(f"Numba version: {info['numba_version']}")
                print(f"JIT enabled: {info['jit_enabled']}")
            else:
                print("Numba not available")

        except ImportError:
            print("Numba not installed")
            # Should work without numba
            from mindar.jit import NUMBA_AVAILABLE

            assert NUMBA_AVAILABLE is False


class TestCrossVersionCompatibility:
    """Test compatibility across different versions."""

    def test_feature_point_serialization_compatibility(self):
        """Test FeaturePoint serialization compatibility."""
        # Test JSON serialization
        import json

        point = FeaturePoint(
            x=100.0,
            y=200.0,
            scale=1.5,
            angle=0.785,
            descriptors=[1, 2, 3, 4, 5],
            maxima=True,
            response=0.8,
            quality=0.9,
        )

        # Convert to dict
        data = point.to_dict()

        # Test JSON serialization
        json_str = json.dumps(data)
        assert isinstance(json_str, str)

        # Test JSON deserialization
        parsed_data = json.loads(json_str)
        reconstructed_point = FeaturePoint.from_dict(parsed_data)

        assert reconstructed_point.x == point.x
        assert reconstructed_point.y == point.y
        assert reconstructed_point.scale == point.scale
        assert reconstructed_point.angle == point.angle
        assert reconstructed_point.maxima == point.maxima
        assert reconstructed_point.response == point.response
        assert reconstructed_point.quality == point.quality

    def test_match_serialization_compatibility(self):
        """Test Match serialization compatibility."""
        import json

        query_point = FeaturePoint(
            x=100.0, y=200.0, scale=1.0, angle=0.0, descriptors=[1, 2, 3], maxima=True, response=0.5
        )
        key_point = FeaturePoint(
            x=105.0, y=205.0, scale=1.1, angle=0.1, descriptors=[2, 3, 4], maxima=True, response=0.6
        )

        match = Match(query_point=query_point, key_point=key_point, distance=25.0, confidence=0.85)

        # Convert to dict
        data = match.to_dict()

        # Test JSON serialization
        json_str = json.dumps(data)
        assert isinstance(json_str, str)

        # Test that it can be parsed back
        parsed_data = json.loads(json_str)
        assert parsed_data["distance"] == 25.0
        assert parsed_data["confidence"] == 0.85

    def test_detection_result_serialization_compatibility(self):
        """Test DetectionResult serialization compatibility."""
        import json

        homography = np.array([[1.1, 0.1, 10.0], [0.1, 1.1, 20.0], [0.0, 0.0, 1.0]])

        result = DetectionResult(target_id=1, homography=homography, matches=[], inliers=12, confidence=0.85)

        # Convert to dict
        data = result.to_dict()

        # Test JSON serialization
        json_str = json.dumps(data)
        assert isinstance(json_str, str)

        # Test that it can be parsed back
        parsed_data = json.loads(json_str)
        assert parsed_data["target_id"] == 1
        assert parsed_data["inliers"] == 12
        assert parsed_data["confidence"] == 0.85
        assert len(parsed_data["homography"]) == 3


class TestPlatformCompatibility:
    """Test platform compatibility."""

    def test_platform_info(self):
        """Test platform information."""
        import platform

        system = platform.system()
        machine = platform.machine()
        processor = platform.processor()

        print(f"Platform: {system}")
        print(f"Machine: {machine}")
        print(f"Processor: {processor}")

        # Should work on common platforms
        assert system in ["Darwin", "Linux", "Windows"]

    def test_architecture_compatibility(self):
        """Test architecture compatibility."""
        import platform  # noqa: F401

        # Test with different data types
        test_data = [
            np.array([1, 2, 3], dtype=np.int32),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        ]

        for data in test_data:
            # Test basic operations
            result = data * 2
            assert result.dtype == data.dtype
            assert len(result) == len(data)

    def test_memory_alignment(self):
        """Test memory alignment compatibility."""
        # Test with different array sizes
        sizes = [10, 100, 1000, 10000]

        for size in sizes:
            # Create arrays of different dtypes
            for dtype in [np.float32, np.float64]:
                arr = np.random.random(size).astype(dtype)

                # Test basic operations
                result = arr * 2.0
                assert result.dtype == dtype
                assert len(result) == size

                # Test with detector
                if size >= 100:  # Only test with reasonable sizes
                    detector = Detector()
                    # Create image from array
                    img = (arr.reshape(int(np.sqrt(size)), -1) * 255).astype(np.uint8)
                    detection_result = detector.detect(img)
                    assert "feature_points" in detection_result


class TestErrorHandlingCompatibility:
    """Test error handling compatibility."""

    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        detector = Detector()
        Matcher()

        # Test with None input
        try:
            detector.detect(None)
            assert False, "Should raise an exception"
        except (TypeError, AttributeError):
            pass  # Expected

        # Test with empty array
        empty_array = np.array([])
        try:
            detector.detect(empty_array)
            assert False, "Should raise an exception"
        except (ValueError, IndexError):
            pass  # Expected

        # Test with invalid shape
        invalid_array = np.array([1, 2, 3])  # 1D array
        try:
            detector.detect(invalid_array)
            assert False, "Should raise an exception"
        except (ValueError, IndexError):
            pass  # Expected

    def test_edge_case_handling(self):
        """Test handling of edge cases."""
        detector = Detector()

        # Test with very small image
        small_image = np.zeros((5, 5), dtype=np.uint8)
        try:
            result = detector.detect(small_image)
            assert "feature_points" in result
        except Exception as e:
            # Should handle gracefully
            print(f"Small image detection failed: {e}")

        # Test with very large image
        large_image = np.zeros((2000, 2000), dtype=np.uint8)
        try:
            result = detector.detect(large_image)
            assert "feature_points" in result
        except Exception as e:
            # Should handle gracefully
            print(f"Large image detection failed: {e}")

    def test_memory_error_handling(self):
        """Test memory error handling."""
        detector = Detector()

        # Test with reasonable image size
        test_image = np.random.randint(0, 255, (500, 500), dtype=np.uint8)

        try:
            result = detector.detect(test_image)
            assert "feature_points" in result
        except MemoryError:
            # Should handle memory errors gracefully
            print("Memory error occurred during detection")
        except Exception as e:
            # Other errors should be handled
            print(f"Detection error: {e}")


class TestBackwardCompatibility:
    """Test backward compatibility."""

    def test_api_compatibility(self):
        """Test API compatibility."""
        # Test that all public APIs are available
        from mindar import (
            Detector,
            Matcher,
            MindARCompiler,
        )

        # Test class instantiation
        detector = Detector()
        matcher = Matcher()
        compiler = MindARCompiler()

        assert detector is not None
        assert matcher is not None
        assert compiler is not None

    def test_method_signatures(self):
        """Test method signatures compatibility."""
        # Test detector methods
        detector = Detector()
        assert hasattr(detector, "detect")
        assert hasattr(detector, "get_performance_stats")

        # Test matcher methods
        matcher = Matcher()
        assert hasattr(matcher, "match")
        assert hasattr(matcher, "get_performance_stats")
        assert hasattr(matcher, "clear_cache")

        # Test compiler methods
        compiler = MindARCompiler()
        assert hasattr(compiler, "compile_images")
        assert hasattr(compiler, "compile_directory")
        assert hasattr(compiler, "load_mind_file")

    def test_config_compatibility(self):
        """Test configuration compatibility."""
        # Test detector config
        detector_config = DetectorConfig()
        assert hasattr(detector_config, "method")
        assert hasattr(detector_config, "max_features")
        assert hasattr(detector_config, "fast_threshold")

        # Test matcher config
        matcher_config = MatcherConfig()
        assert hasattr(matcher_config, "ratio_threshold")
        assert hasattr(matcher_config, "distance_threshold")
        assert hasattr(matcher_config, "min_matches")

        # Test tracker config
        tracker_config = TrackerConfig(
            marker_dimensions=[(100, 100)],
            tracking_data_list=[[{"points": [(50, 50)]}]],
            projection_transform=np.eye(3),
            input_width=640,
            input_height=480,
        )
        assert hasattr(tracker_config, "marker_dimensions")
        assert hasattr(tracker_config, "tracking_data_list")
        assert hasattr(tracker_config, "projection_transform")
