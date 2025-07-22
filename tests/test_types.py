"""
Tests for MindAR data types and structures.
Ensures compatibility across Python versions and proper serialization.
"""

import json
import pickle
from typing import Dict, List

import numpy as np
import pytest

from mindar.types import DetectionResult, FeaturePoint, Match


class TestFeaturePoint:
    """Test FeaturePoint data structure."""

    def test_feature_point_creation(self):
        """Test basic FeaturePoint creation."""
        descriptors = [1, 2, 3, 4, 5] * 12  # 60 descriptors
        point = FeaturePoint(
            x=100.5, y=200.3, scale=1.5, angle=0.785, descriptors=descriptors, maxima=True, response=0.8, quality=0.9
        )

        assert point.x == 100.5
        assert point.y == 200.3
        assert point.scale == 1.5
        assert point.angle == 0.785
        assert point.descriptors == descriptors
        assert point.maxima is True
        assert point.response == 0.8
        assert point.quality == 0.9

    def test_feature_point_default_quality(self):
        """Test FeaturePoint with default quality."""
        point = FeaturePoint(x=100.0, y=200.0, scale=1.0, angle=0.0, descriptors=[1, 2, 3], maxima=True, response=0.5)

        assert point.quality == 0.5  # Default value

    def test_feature_point_get_position(self):
        """Test get_position method."""
        point = FeaturePoint(x=150.0, y=250.0, scale=1.0, angle=0.0, descriptors=[1, 2, 3], maxima=True, response=0.5)

        position = point.get_position()
        assert position == (150.0, 250.0)
        assert isinstance(position, tuple)

    def test_feature_point_to_dict(self):
        """Test serialization to dictionary."""
        descriptors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        point = FeaturePoint(
            x=100.0, y=200.0, scale=1.5, angle=0.5, descriptors=descriptors, maxima=False, response=0.7, quality=0.8
        )

        data = point.to_dict()

        assert data["x"] == 100.0
        assert data["y"] == 200.0
        assert data["scale"] == 1.5
        assert data["angle"] == 0.5
        assert data["maxima"] is False
        assert data["response"] == 0.7
        assert data["quality"] == 0.8
        assert len(data["descriptors"]) == 8  # Should truncate to 8
        assert data["descriptors"] == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_feature_point_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "x": 150.0,
            "y": 250.0,
            "scale": 2.0,
            "angle": 1.0,
            "maxima": True,
            "response": 0.9,
            "descriptors": [1, 2, 3, 4],
            "quality": 0.95,
        }

        point = FeaturePoint.from_dict(data)

        assert point.x == 150.0
        assert point.y == 250.0
        assert point.scale == 2.0
        assert point.angle == 1.0
        assert point.maxima is True
        assert point.response == 0.9
        assert point.quality == 0.95
        assert point.descriptors == [1, 2, 3, 4]

    def test_feature_point_json_serialization(self):
        """Test JSON serialization compatibility."""
        point = FeaturePoint(
            x=100.0, y=200.0, scale=1.0, angle=0.0, descriptors=[1, 2, 3, 4], maxima=True, response=0.5, quality=0.7
        )

        # Convert to dict and then to JSON
        data = point.to_dict()
        json_str = json.dumps(data)

        # Parse back
        parsed_data = json.loads(json_str)
        reconstructed_point = FeaturePoint.from_dict(parsed_data)

        assert reconstructed_point.x == point.x
        assert reconstructed_point.y == point.y
        assert reconstructed_point.scale == point.scale
        assert reconstructed_point.angle == point.angle
        assert reconstructed_point.maxima == point.maxima
        assert reconstructed_point.response == point.response
        assert reconstructed_point.quality == point.quality


class TestMatch:
    """Test Match data structure."""

    def test_match_creation(self):
        """Test basic Match creation."""
        query_point = FeaturePoint(
            x=100.0, y=200.0, scale=1.0, angle=0.0, descriptors=[1, 2, 3], maxima=True, response=0.5
        )
        key_point = FeaturePoint(
            x=105.0, y=205.0, scale=1.1, angle=0.1, descriptors=[2, 3, 4], maxima=True, response=0.6
        )

        match = Match(query_point=query_point, key_point=key_point, distance=25.0, confidence=0.85)

        assert match.query_point == query_point
        assert match.key_point == key_point
        assert match.distance == 25.0
        assert match.confidence == 0.85

    def test_match_default_confidence(self):
        """Test Match with default confidence calculation."""
        query_point = FeaturePoint(
            x=100.0, y=200.0, scale=1.0, angle=0.0, descriptors=[1, 2, 3], maxima=True, response=0.5
        )
        key_point = FeaturePoint(
            x=105.0, y=205.0, scale=1.1, angle=0.1, descriptors=[2, 3, 4], maxima=True, response=0.6
        )

        match = Match(query_point=query_point, key_point=key_point, distance=50.0)

        # Confidence should be calculated as max(0.0, 1.0 - distance / 100.0)
        expected_confidence = max(0.0, 1.0 - 50.0 / 100.0)
        assert match.confidence == expected_confidence

    def test_match_to_dict(self):
        """Test Match serialization to dictionary."""
        query_point = FeaturePoint(
            x=100.0, y=200.0, scale=1.0, angle=0.0, descriptors=[1, 2, 3], maxima=True, response=0.5
        )
        key_point = FeaturePoint(
            x=105.0, y=205.0, scale=1.1, angle=0.1, descriptors=[2, 3, 4], maxima=True, response=0.6
        )

        match = Match(query_point=query_point, key_point=key_point, distance=30.0, confidence=0.8)

        data = match.to_dict()

        assert data["distance"] == 30.0
        assert data["confidence"] == 0.8
        assert "query_point" in data
        assert "key_point" in data
        assert data["query_point"]["x"] == 100.0
        assert data["key_point"]["x"] == 105.0


class TestDetectionResult:
    """Test DetectionResult data structure."""

    def test_detection_result_creation(self):
        """Test basic DetectionResult creation."""
        homography = np.eye(3)
        matches = []

        result = DetectionResult(target_id=1, homography=homography, matches=matches, inliers=10, confidence=0.9)

        assert result.target_id == 1
        assert np.array_equal(result.homography, homography)
        assert result.matches == matches
        assert result.inliers == 10
        assert result.confidence == 0.9
        assert result.target_image is None

    def test_detection_result_is_valid(self):
        """Test is_valid method."""
        homography = np.eye(3)
        matches = []

        # Valid result
        result = DetectionResult(target_id=1, homography=homography, matches=matches, inliers=8, confidence=0.7)
        assert result.is_valid() is True

        # Invalid result - too few inliers
        result.inliers = 4
        assert result.is_valid() is False

        # Invalid result - low confidence
        result.inliers = 8
        result.confidence = 0.3
        assert result.is_valid() is False

        # Custom thresholds
        result.confidence = 0.7
        assert result.is_valid(min_inliers=10, min_confidence=0.6) is False
        assert result.is_valid(min_inliers=6, min_confidence=0.8) is False

    def test_detection_result_to_dict(self):
        """Test DetectionResult serialization to dictionary."""
        homography = np.array([[1.1, 0.1, 10.0], [0.1, 1.1, 20.0], [0.0, 0.0, 1.0]])

        # Create some matches
        query_point = FeaturePoint(
            x=100.0, y=200.0, scale=1.0, angle=0.0, descriptors=[1, 2, 3], maxima=True, response=0.5
        )
        key_point = FeaturePoint(
            x=105.0, y=205.0, scale=1.1, angle=0.1, descriptors=[2, 3, 4], maxima=True, response=0.6
        )
        match = Match(query_point=query_point, key_point=key_point, distance=25.0, confidence=0.8)

        result = DetectionResult(
            target_id=2, homography=homography, matches=[match] * 15, inliers=12, confidence=0.85  # 15 matches
        )

        data = result.to_dict()

        assert data["target_id"] == 2
        assert data["inliers"] == 12
        assert data["confidence"] == 0.85
        assert data["matches_count"] == 15
        assert len(data["matches"]) == 10  # Should truncate to 10
        assert len(data["homography"]) == 3
        assert len(data["homography"][0]) == 3


class TestTypesCompatibility:
    """Test type compatibility across Python versions."""

    def test_feature_point_pickle_compatibility(self):
        """Test pickle serialization compatibility."""
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

        # Pickle and unpickle
        pickled = pickle.dumps(point)
        unpickled = pickle.loads(pickled)

        assert unpickled.x == point.x
        assert unpickled.y == point.y
        assert unpickled.scale == point.scale
        assert unpickled.angle == point.angle
        assert unpickled.descriptors == point.descriptors
        assert unpickled.maxima == point.maxima
        assert unpickled.response == point.response
        assert unpickled.quality == point.quality

    def test_match_pickle_compatibility(self):
        """Test Match pickle serialization compatibility."""
        query_point = FeaturePoint(
            x=100.0, y=200.0, scale=1.0, angle=0.0, descriptors=[1, 2, 3], maxima=True, response=0.5
        )
        key_point = FeaturePoint(
            x=105.0, y=205.0, scale=1.1, angle=0.1, descriptors=[2, 3, 4], maxima=True, response=0.6
        )

        match = Match(query_point=query_point, key_point=key_point, distance=25.0, confidence=0.85)

        # Pickle and unpickle
        pickled = pickle.dumps(match)
        unpickled = pickle.loads(pickled)

        assert unpickled.distance == match.distance
        assert unpickled.confidence == match.confidence
        assert unpickled.query_point.x == match.query_point.x
        assert unpickled.key_point.x == match.key_point.x

    def test_numpy_compatibility(self):
        """Test numpy array compatibility in DetectionResult."""
        # Test with different numpy dtypes
        for dtype in [np.float32, np.float64]:
            homography = np.eye(3, dtype=dtype)

            result = DetectionResult(target_id=1, homography=homography, matches=[], inliers=10, confidence=0.9)

            # Should work with any numpy dtype
            assert result.is_valid()
            data = result.to_dict()
            assert len(data["homography"]) == 3

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with empty descriptors
        point = FeaturePoint(x=0.0, y=0.0, scale=0.0, angle=0.0, descriptors=[], maxima=True, response=0.0, quality=0.0)

        assert point.x == 0.0
        assert point.y == 0.0
        assert point.descriptors == []

        # Test with very large values
        point = FeaturePoint(
            x=1e6, y=1e6, scale=1e3, angle=2 * np.pi, descriptors=[255] * 100, maxima=False, response=1.0, quality=1.0
        )

        assert point.x == 1e6
        assert point.y == 1e6
        assert point.scale == 1e3
        assert point.angle == 2 * np.pi
        assert len(point.descriptors) == 100
