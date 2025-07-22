"""
Tests for MindAR matcher module.
Tests feature matching algorithms and cross-version compatibility.
"""

import time

import numpy as np
import pytest

from mindar.matcher import ClusterNode, Match, Matcher, MatcherConfig
from mindar.types import FeaturePoint


class TestMatcherConfig:
    """Test MatcherConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MatcherConfig()

        assert config.ratio_threshold == 0.75
        assert config.distance_threshold == 0.8
        assert config.min_matches == 8
        assert config.ransac_threshold == 3.0
        assert config.debug_mode is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MatcherConfig(
            ratio_threshold=0.8, distance_threshold=0.7, min_matches=10, ransac_threshold=2.5, debug_mode=True
        )

        assert config.ratio_threshold == 0.8
        assert config.distance_threshold == 0.7
        assert config.min_matches == 10
        assert config.ransac_threshold == 2.5
        assert config.debug_mode is True


class TestMatch:
    """Test Match data structure."""

    def test_match_creation(self):
        """Test basic Match creation."""
        query_point = FeaturePoint(
            x=100.0, y=200.0, scale=1.0, angle=0.0, descriptors=[1, 2, 3], maxima=True, response=0.5
        )
        train_point = FeaturePoint(
            x=105.0, y=205.0, scale=1.1, angle=0.1, descriptors=[2, 3, 4], maxima=True, response=0.6
        )

        match = Match(query_idx=0, train_idx=1, distance=25.0, query_point=query_point, train_point=train_point)

        assert match.query_idx == 0
        assert match.train_idx == 1
        assert match.distance == 25.0
        assert match.query_point == query_point
        assert match.train_point == train_point


class TestClusterNode:
    """Test ClusterNode data structure."""

    def test_cluster_node_creation(self):
        """Test basic ClusterNode creation."""
        node = ClusterNode(center=(100.0, 200.0), radius=50.0, children=[], indices=[0, 1, 2], depth=1)

        assert node.center == (100.0, 200.0)
        assert node.radius == 50.0
        assert node.children == []
        assert node.indices == [0, 1, 2]
        assert node.depth == 1

    def test_cluster_node_defaults(self):
        """Test ClusterNode with default values."""
        node = ClusterNode(center=(0.0, 0.0), radius=0.0)

        assert node.children == []
        assert node.indices == []
        assert node.depth == 0


class TestMatcher:
    """Test Matcher class functionality."""

    def test_matcher_initialization(self):
        """Test matcher initialization."""
        config = MatcherConfig(debug_mode=True)
        matcher = Matcher(config)

        assert matcher.config == config
        assert len(matcher.matching_times) == 0
        assert hasattr(matcher, "cached_clusters")
        assert hasattr(matcher, "cached_descriptors")

    def test_matcher_default_initialization(self):
        """Test matcher with default config."""
        matcher = Matcher()

        assert matcher.config.ratio_threshold == 0.75
        assert matcher.config.distance_threshold == 0.8
        assert matcher.config.min_matches == 8

    def test_match_empty_feature_sets(self):
        """Test matching with empty feature sets."""
        matcher = Matcher()

        # Both empty
        result = matcher.match([], [])
        assert result["matches"] == []
        assert result["homography"] is None
        assert result["inliers"] == []

        # One empty
        features = [
            FeaturePoint(x=100.0, y=200.0, scale=1.0, angle=0.0, descriptors=[1, 2, 3], maxima=True, response=0.5)
        ]

        result = matcher.match(features, [])
        assert result["matches"] == []
        assert result["homography"] is None
        assert result["inliers"] == []

        result = matcher.match([], features)
        assert result["matches"] == []
        assert result["homography"] is None
        assert result["inliers"] == []

    def test_match_single_features(self, feature_points):
        """Test matching with single features."""
        matcher = Matcher()

        if len(feature_points) >= 2:
            features1 = [feature_points[0]]
            features2 = [feature_points[1]]

            result = matcher.match(features1, features2)

            assert "matches" in result
            assert "homography" in result
            assert "inliers" in result
            assert isinstance(result["matches"], list)
            assert isinstance(result["inliers"], list)

    def test_match_multiple_features(self, feature_points):
        """Test matching with multiple features."""
        matcher = Matcher()

        if len(feature_points) >= 4:
            features1 = feature_points[:2]
            features2 = feature_points[2:4]

            result = matcher.match(features1, features2)

            assert "matches" in result
            assert "homography" in result
            assert "inliers" in result
            assert isinstance(result["matches"], list)
            assert isinstance(result["inliers"], list)

    def test_hierarchical_clustering(self, feature_points):
        """Test hierarchical clustering functionality."""
        matcher = Matcher()

        if len(feature_points) >= 3:
            cluster_tree = matcher._build_hierarchical_clusters(feature_points)

            assert isinstance(cluster_tree, ClusterNode)
            assert hasattr(cluster_tree, "center")
            assert hasattr(cluster_tree, "radius")
            assert hasattr(cluster_tree, "children")
            assert hasattr(cluster_tree, "indices")
            assert hasattr(cluster_tree, "depth")

    def test_cluster_query(self, feature_points):
        """Test cluster tree query functionality."""
        matcher = Matcher()

        if len(feature_points) >= 3:
            cluster_tree = matcher._build_hierarchical_clusters(feature_points)
            query_point = (100.0, 200.0)

            candidates = matcher._query_cluster_tree(cluster_tree, query_point, feature_points, max_candidates=5)

            assert isinstance(candidates, list)
            assert all(isinstance(idx, int) for idx in candidates)
            assert all(0 <= idx < len(feature_points) for idx in candidates)

    def test_brute_force_matching(self, feature_points):
        """Test brute force matching."""
        matcher = Matcher()

        if len(feature_points) >= 4:
            features1 = feature_points[:2]
            features2 = feature_points[2:4]

            matches = matcher._match_brute_force(features1, features2)

            assert isinstance(matches, list)
            assert len(matches) == len(features1)

            for match_list in matches:
                assert isinstance(match_list, list)
                for match in match_list:
                    assert isinstance(match, Match)
                    assert hasattr(match, "query_idx")
                    assert hasattr(match, "train_idx")
                    assert hasattr(match, "distance")

    def test_descriptor_distance_computation(self):
        """Test descriptor distance computation."""
        matcher = Matcher()

        # Test with identical descriptors
        desc1 = [1, 2, 3, 4, 5]
        desc2 = [1, 2, 3, 4, 5]
        distance = matcher._compute_descriptor_distance(desc1, desc2)
        assert distance == 0.0

        # Test with different descriptors
        desc3 = [6, 7, 8, 9, 10]
        distance = matcher._compute_descriptor_distance(desc1, desc3)
        assert distance > 0.0

        # Test with different lengths (should handle gracefully)
        desc4 = [1, 2, 3]
        distance = matcher._compute_descriptor_distance(desc1, desc4)
        assert isinstance(distance, float)

    def test_ratio_test(self, feature_points):
        """Test Lowe's ratio test."""
        matcher = Matcher()

        if len(feature_points) >= 4:
            features1 = feature_points[:2]
            features2 = feature_points[2:4]

            # Create some matches
            matches_per_query = []
            for i, feature1 in enumerate(features1):
                query_matches = []
                for j, feature2 in enumerate(features2):
                    match = Match(
                        query_idx=i,
                        train_idx=j,
                        distance=float(abs(i - j) * 10),
                        query_point=feature1,
                        train_point=feature2,
                    )
                    query_matches.append(match)
                matches_per_query.append(query_matches)

            good_matches = matcher._apply_ratio_test(matches_per_query)

            assert isinstance(good_matches, list)
            for match in good_matches:
                assert isinstance(match, Match)

    def test_homography_estimation(self, feature_points):
        """Test homography estimation."""
        matcher = Matcher()

        if len(feature_points) >= 8:
            # Create matches with known geometric relationship
            matches = []
            for i in range(min(8, len(feature_points) // 2)):
                query_point = feature_points[i]
                train_point = feature_points[i + len(feature_points) // 2]

                match = Match(
                    query_idx=i,
                    train_idx=i + len(feature_points) // 2,
                    distance=float(i * 5),
                    query_point=query_point,
                    train_point=train_point,
                )
                matches.append(match)

            homography, inliers = matcher._estimate_homography(matches)

            # Should return either a valid homography or None
            if homography is not None:
                assert homography.shape == (3, 3)
                assert isinstance(inliers, list)
                assert all(isinstance(match, Match) for match in inliers)
            else:
                assert inliers == []

    def test_performance_stats(self, feature_points):
        """Test performance statistics collection."""
        matcher = Matcher()

        if len(feature_points) >= 4:
            features1 = feature_points[:2]
            features2 = feature_points[2:4]

            # Run matching multiple times
            for _ in range(3):
                matcher.match(features1, features2)

            stats = matcher.get_performance_stats()

            assert "avg_matching_time" in stats
            assert "min_matching_time" in stats
            assert "max_matching_time" in stats
            assert "total_matches" in stats
            assert stats["total_matches"] == 3
            assert stats["avg_matching_time"] > 0.0

    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        matcher = Matcher()

        # Add some dummy cache entries
        matcher.cached_clusters[123] = "dummy_cluster"
        matcher.cached_descriptors[456] = "dummy_descriptor"

        assert len(matcher.cached_clusters) > 0
        assert len(matcher.cached_descriptors) > 0

        matcher.clear_cache()

        assert len(matcher.cached_clusters) == 0
        assert len(matcher.cached_descriptors) == 0


class TestMatcherCompatibility:
    """Test matcher compatibility across different scenarios."""

    def test_different_feature_counts(self, feature_points):
        """Test matching with different feature counts."""
        matcher = Matcher()

        if len(feature_points) >= 6:
            # Test different combinations
            combinations = [(1, 1), (1, 5), (5, 1), (2, 3), (3, 2), (5, 5)]

            for count1, count2 in combinations:
                features1 = feature_points[:count1]
                features2 = feature_points[count1 : count1 + count2]

                result = matcher.match(features1, features2)

                assert "matches" in result
                assert "homography" in result
                assert "inliers" in result

    def test_different_descriptor_lengths(self):
        """Test matching with different descriptor lengths."""
        matcher = Matcher()

        # Create features with different descriptor lengths
        feature1 = FeaturePoint(
            x=100.0, y=200.0, scale=1.0, angle=0.0, descriptors=[1, 2, 3], maxima=True, response=0.5
        )
        feature2 = FeaturePoint(
            x=105.0, y=205.0, scale=1.1, angle=0.1, descriptors=[2, 3, 4, 5, 6], maxima=True, response=0.6
        )

        result = matcher.match([feature1], [feature2])

        assert "matches" in result
        assert "homography" in result
        assert "inliers" in result

    def test_matching_quality_assessment(self, feature_points):
        """Test matching quality assessment."""
        matcher = Matcher()

        if len(feature_points) >= 4:
            features1 = feature_points[:2]
            features2 = feature_points[2:4]

            result = matcher.match(features1, features2)

            # Check match quality
            for match in result["matches"]:
                assert match.distance >= 0.0
                assert isinstance(match.distance, float)

                # Check indices are valid
                assert 0 <= match.query_idx < len(features1)
                assert 0 <= match.train_idx < len(features2)

    def test_geometric_consistency(self, feature_points):
        """Test geometric consistency of matches."""
        matcher = Matcher()

        if len(feature_points) >= 8:
            features1 = feature_points[:4]
            features2 = feature_points[4:8]

            result = matcher.match(features1, features2)

            if result["homography"] is not None and len(result["inliers"]) > 0:
                # Check that homography is a 3x3 matrix
                homography = result["homography"]
                assert homography.shape == (3, 3)
                assert homography.dtype in [np.float32, np.float64]

                # Check that inliers are valid matches
                for inlier in result["inliers"]:
                    assert isinstance(inlier, Match)
                    assert inlier in result["matches"]


@pytest.mark.performance
class TestMatcherPerformance:
    """Performance tests for matcher."""

    def test_matching_performance(self, feature_points):
        """Test matching performance."""
        matcher = Matcher()

        if len(feature_points) >= 20:
            features1 = feature_points[:10]
            features2 = feature_points[10:20]

            # Warm up
            matcher.match(features1, features2)

            # Performance test
            start_time = time.time()
            for _ in range(10):
                matcher.match(features1, features2)
            total_time = time.time() - start_time

            avg_time = total_time / 10
            assert avg_time < 1.0  # Should be under 1 second per match

    def test_clustering_performance(self, feature_points):
        """Test clustering performance."""
        matcher = Matcher()

        if len(feature_points) >= 50:
            # Test clustering with larger feature sets
            start_time = time.time()
            cluster_tree = matcher._build_hierarchical_clusters(feature_points)
            clustering_time = time.time() - start_time

            assert clustering_time < 0.1  # Should be under 100ms
            assert isinstance(cluster_tree, ClusterNode)

    def test_cache_performance(self, feature_points):
        """Test caching performance benefits."""
        matcher = Matcher()

        if len(feature_points) >= 10:
            features1 = feature_points[:5]
            features2 = feature_points[5:10]

            # First match (no cache)
            start_time = time.time()
            matcher.match(features1, features2)
            time1 = time.time() - start_time

            # Second match (with cache)
            start_time = time.time()
            matcher.match(features1, features2)
            time2 = time.time() - start_time

            # Cached match should be faster
            assert time2 <= time1
            assert len(matcher.cached_clusters) > 0


class TestMatcherEdgeCases:
    """Test matcher edge cases and error handling."""

    def test_identical_features(self):
        """Test matching identical features."""
        matcher = Matcher()

        # Create identical features
        feature = FeaturePoint(
            x=100.0, y=200.0, scale=1.0, angle=0.0, descriptors=[1, 2, 3, 4, 5], maxima=True, response=0.5
        )

        result = matcher.match([feature], [feature])

        assert "matches" in result
        assert "homography" in result
        assert "inliers" in result

    def test_single_descriptor_values(self):
        """Test matching with single descriptor values."""
        matcher = Matcher()

        feature1 = FeaturePoint(x=100.0, y=200.0, scale=1.0, angle=0.0, descriptors=[0], maxima=True, response=0.5)
        feature2 = FeaturePoint(x=105.0, y=205.0, scale=1.1, angle=0.1, descriptors=[255], maxima=True, response=0.6)

        result = matcher.match([feature1], [feature2])

        assert "matches" in result
        assert "homography" in result
        assert "inliers" in result

    def test_extreme_coordinates(self):
        """Test matching with extreme coordinate values."""
        matcher = Matcher()

        feature1 = FeaturePoint(x=1e6, y=1e6, scale=1.0, angle=0.0, descriptors=[1, 2, 3], maxima=True, response=0.5)
        feature2 = FeaturePoint(x=-1e6, y=-1e6, scale=1.1, angle=0.1, descriptors=[4, 5, 6], maxima=True, response=0.6)

        result = matcher.match([feature1], [feature2])

        assert "matches" in result
        assert "homography" in result
        assert "inliers" in result
