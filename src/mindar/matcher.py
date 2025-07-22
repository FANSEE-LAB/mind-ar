"""
MindAR-compatible matcher.

This module provides high-performance feature matching algorithms,
including binary descriptor matching with Hamming distance computation.
Compatible with MindAR's matching pipeline and optimized for edge devices.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .jit import NUMBA_AVAILABLE, _hamming_distance_jit_impl
from .types import FeaturePoint

# Constants optimized for real-time performance
DEFAULT_RATIO_THRESHOLD = 0.75
DEFAULT_DISTANCE_THRESHOLD = 0.8
DEFAULT_MIN_MATCHES = 8
DEFAULT_RANSAC_THRESHOLD = 3.0
DEFAULT_MAX_MATCHES = 100

# Hierarchical clustering parameters
CLUSTER_MAX_POP = 8
HOUGH_BINS = 180
HOUGH_SCALE_BINS = 10

# Performance optimization flags
ENABLE_HIERARCHICAL_CLUSTERING = True
ENABLE_LOWE_RATIO_TEST = True
ENABLE_GEOMETRIC_VERIFICATION = True
ENABLE_DISTANCE_THRESHOLD = True


@dataclass
class MatcherConfig:
    """Configuration for the matcher to reduce parameter count."""

    ratio_threshold: float = DEFAULT_RATIO_THRESHOLD
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD
    min_matches: int = DEFAULT_MIN_MATCHES
    ransac_threshold: float = DEFAULT_RANSAC_THRESHOLD
    debug_mode: bool = False


@dataclass
class Match:
    """
    Represents a match between two feature points.

    Attributes:
        query_idx: Index of the feature in the query set
        train_idx: Index of the feature in the train set
        distance: Distance between the features (lower is better)
        query_point: Query feature point
        train_point: Train feature point
    """

    query_idx: int
    train_idx: int
    distance: float
    query_point: Optional[FeaturePoint] = None
    train_point: Optional[FeaturePoint] = None


@dataclass
class ClusterNode:
    """
    Node in hierarchical clustering tree.

    Attributes:
        center: Center point of cluster
        radius: Radius of cluster
        children: Child nodes
        indices: Feature indices in this cluster
        depth: Depth in tree
    """

    center: Tuple[float, float]
    radius: float
    children: List["ClusterNode"] = None
    indices: List[int] = None
    depth: int = 0

    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.indices is None:
            self.indices = []


class Matcher:
    """
    High-performance feature matcher optimized for edge devices.

    Uses hierarchical clustering for fast spatial queries and
    optimized Hamming distance computation for binary descriptors.
    """

    def __init__(self, config: MatcherConfig = None):
        """
        Initialize optimized matcher.

        Args:
            config: Matcher configuration object
        """
        if config is None:
            config = MatcherConfig()

        self.config = config

        # Performance tracking
        self.matching_times = []

        # Cached structures for repeated matching
        self.cached_clusters = {}
        self.cached_descriptors = {}

    def match(self, features1: List[FeaturePoint], features2: List[FeaturePoint]) -> Dict:
        """
        Match two feature sets with hierarchical clustering optimization.

        Args:
            features1: First feature set (template/keyframe)
            features2: Second feature set (query)

        Returns:
            Dictionary with matches, homography, and inliers
        """
        start_time = time.time()

        if not features1 or not features2:
            return {"matches": [], "homography": None, "inliers": []}

        # Build hierarchical clusters for features1 if not cached
        cluster_key = id(features1)
        if ENABLE_HIERARCHICAL_CLUSTERING and cluster_key not in self.cached_clusters:
            self.cached_clusters[cluster_key] = self._build_hierarchical_clusters(features1)

        # Find initial matches using fast spatial search
        if ENABLE_HIERARCHICAL_CLUSTERING and cluster_key in self.cached_clusters:
            raw_matches = self._match_with_clustering(features1, features2, self.cached_clusters[cluster_key])
        else:
            raw_matches = self._match_brute_force(features1, features2)

        # Apply ratio test
        if ENABLE_LOWE_RATIO_TEST:
            good_matches = self._apply_ratio_test(raw_matches)
        else:
            good_matches = raw_matches

        # Geometric verification and homography estimation
        homography = None
        inliers = []

        if len(good_matches) >= self.config.min_matches and ENABLE_GEOMETRIC_VERIFICATION:
            homography, inliers = self._estimate_homography(good_matches)

        # Track performance
        matching_time = time.time() - start_time
        self.matching_times.append(matching_time)

        if self.config.debug_mode:
            print(f"Matched {len(good_matches)} features in {matching_time*1000:.1f}ms")
            if homography is not None:
                print(f"Found homography with {len(inliers)} inliers")

        return {"matches": good_matches, "homography": homography, "inliers": inliers}

    def _build_hierarchical_clusters(self, features: List[FeaturePoint]) -> ClusterNode:
        """
        Build hierarchical clustering tree for fast spatial queries.

        Args:
            features: Features to cluster

        Returns:
            Root node of clustering tree
        """
        if not features:
            return ClusterNode(center=(0, 0), radius=0)

        # Extract positions
        positions = np.array([[feature.x, feature.y] for feature in features])
        indices = list(range(len(features)))

        # Recursively build tree
        return self._build_cluster_recursive(positions, indices, depth=0)

    def _build_cluster_recursive(
        self, positions: np.ndarray, indices: List[int], depth: int, max_depth: int = 10
    ) -> ClusterNode:
        """Recursively build clustering tree."""
        if len(indices) <= CLUSTER_MAX_POP or depth >= max_depth:
            # Leaf node
            center = np.mean(positions[indices], axis=0)
            if len(indices) > 1:
                distances = np.linalg.norm(positions[indices] - center, axis=1)
                radius = np.max(distances)
            else:
                radius = 0.0

            return ClusterNode(center=tuple(center), radius=float(radius), indices=indices, depth=depth)

        # Split along longest dimension
        coords = positions[indices]
        ranges = np.max(coords, axis=0) - np.min(coords, axis=0)
        split_dim = np.argmax(ranges)

        # Sort by split dimension
        sorted_indices = sorted(indices, key=lambda i: positions[i][split_dim])
        mid = len(sorted_indices) // 2

        # Create child clusters
        left_indices = sorted_indices[:mid]
        right_indices = sorted_indices[mid:]

        left_child = self._build_cluster_recursive(positions, left_indices, depth + 1, max_depth)
        right_child = self._build_cluster_recursive(positions, right_indices, depth + 1, max_depth)

        # Create parent node
        center = np.mean(positions[indices], axis=0)
        distances = np.linalg.norm(positions[indices] - center, axis=1)
        radius = np.max(distances)

        return ClusterNode(center=tuple(center), radius=float(radius), children=[left_child, right_child], depth=depth)

    def _match_with_clustering(
        self, features1: List[FeaturePoint], features2: List[FeaturePoint], cluster_tree: ClusterNode
    ) -> List[List[Match]]:
        """
        Match features using hierarchical clustering for spatial acceleration.

        Args:
            features1: Template features
            features2: Query features
            cluster_tree: Hierarchical clustering tree for features1

        Returns:
            List of matches for each query feature
        """
        matches_per_query = []

        for query_idx, query_feature in enumerate(features2):
            # Find candidate features using spatial clustering
            candidates = self._query_cluster_tree(cluster_tree, (query_feature.x, query_feature.y), features1)

            # Compute descriptor distances for candidates
            feature_matches = []
            for candidate_idx in candidates:
                if candidate_idx < len(features1):
                    candidate_feature = features1[candidate_idx]

                    # Skip if different extrema type (maxima vs minima)
                    if query_feature.maxima != candidate_feature.maxima:
                        continue

                    distance = self._compute_descriptor_distance(
                        query_feature.descriptors, candidate_feature.descriptors
                    )

                    if ENABLE_DISTANCE_THRESHOLD and distance > self.config.distance_threshold:
                        continue

                    feature_matches.append(
                        Match(
                            query_idx=query_idx,
                            train_idx=candidate_idx,
                            distance=distance,
                            query_point=query_feature,
                            train_point=candidate_feature,
                        )
                    )

            # Sort by distance and keep top matches
            feature_matches.sort(key=lambda m: m.distance)
            matches_per_query.append(feature_matches[:2])  # Keep top 2 for ratio test

        return matches_per_query

    def _query_cluster_tree(
        self,
        node: ClusterNode,
        query_point: Tuple[float, float],
        features: List[FeaturePoint],
        max_candidates: int = 20,
    ) -> List[int]:
        """
        Query clustering tree for candidate features near query point.

        Args:
            node: Current cluster node
            query_point: Query point coordinates
            features: Original feature list
            max_candidates: Maximum candidates to return

        Returns:
            List of candidate feature indices
        """
        candidates = []

        # Check if query point is within cluster radius
        center_dist = np.linalg.norm([query_point[0] - node.center[0], query_point[1] - node.center[1]])

        if center_dist > node.radius * 2.0:  # Outside cluster
            return candidates

        if node.indices:  # Leaf node
            candidates.extend(node.indices)
        else:  # Internal node
            for child in node.children:
                child_candidates = self._query_cluster_tree(child, query_point, features, max_candidates)
                candidates.extend(child_candidates)

        # Limit candidates and sort by distance
        if len(candidates) > max_candidates:
            distances = []
            for idx in candidates:
                if idx < len(features):
                    feat = features[idx]
                    dist = np.linalg.norm([query_point[0] - feat.x, query_point[1] - feat.y])
                    distances.append((dist, idx))

            distances.sort()
            candidates = [idx for _, idx in distances[:max_candidates]]

        return candidates

    def _match_brute_force(self, features1: List[FeaturePoint], features2: List[FeaturePoint]) -> List[List[Match]]:
        """
        Brute force matching for small feature sets or fallback.

        Args:
            features1: Template features
            features2: Query features

        Returns:
            List of matches for each query feature
        """
        matches_per_query = []

        for query_idx, query_feature in enumerate(features2):
            feature_matches = []

            for train_idx, train_feature in enumerate(features1):
                # Skip if different extrema type
                if query_feature.maxima != train_feature.maxima:
                    continue

                distance = self._compute_descriptor_distance(query_feature.descriptors, train_feature.descriptors)

                if ENABLE_DISTANCE_THRESHOLD and distance > self.config.distance_threshold:
                    continue

                feature_matches.append(
                    Match(
                        query_idx=query_idx,
                        train_idx=train_idx,
                        distance=distance,
                        query_point=query_feature,
                        train_point=train_feature,
                    )
                )

            # Sort by distance and keep top matches
            feature_matches.sort(key=lambda m: m.distance)
            matches_per_query.append(feature_matches[:2])  # Keep top 2 for ratio test

        return matches_per_query

    def _compute_hamming_distance_jit(self, desc1: np.ndarray, desc2: np.ndarray) -> int:
        """
        Fast Hamming distance computation.

        Uses JIT compilation if numba is available, otherwise falls back to pure Python.

        Args:
            desc1: First descriptor array
            desc2: Second descriptor array

        Returns:
            Hamming distance between descriptors
        """
        return _hamming_distance_jit_impl(desc1, desc2)

    def _compute_descriptor_distance(self, desc1: List[int], desc2: List[int]) -> float:
        """
        Compute distance between binary descriptors.

        Args:
            desc1: First descriptor
            desc2: Second descriptor

        Returns:
            Normalized distance [0, 1]
        """
        if not desc1 or not desc2:
            return 1.0

        if NUMBA_AVAILABLE:
            # Use fast Numba implementation - pass lists directly to avoid type inference issues
            hamming_dist = self._compute_hamming_distance_jit(desc1, desc2)
        else:
            # Fallback to Python implementation
            hamming_dist = 0
            for d1, d2 in zip(desc1, desc2):
                xor = d1 ^ d2
                while xor:
                    hamming_dist += 1
                    xor &= xor - 1

        # Normalize by maximum possible distance
        max_distance = min(len(desc1), len(desc2)) * 32  # 32 bits per integer
        return hamming_dist / max_distance if max_distance > 0 else 1.0

    def _apply_ratio_test(self, matches_per_query: List[List[Match]]) -> List[Match]:
        """
        Apply Lowe's ratio test to filter matches.

        Args:
            matches_per_query: Raw matches for each query feature

        Returns:
            Filtered good matches
        """
        good_matches = []

        for feature_matches in matches_per_query:
            if len(feature_matches) >= 2:
                best_match = feature_matches[0]
                second_match = feature_matches[1]

                # Apply ratio test
                if best_match.distance < self.config.ratio_threshold * second_match.distance:
                    good_matches.append(best_match)
            elif len(feature_matches) == 1:
                # Only one match found
                good_matches.append(feature_matches[0])

        return good_matches

    def _estimate_homography(self, matches: List[Match]) -> Tuple[Optional[np.ndarray], List[Match]]:
        """
        Estimate homography using RANSAC and return inliers.

        Args:
            matches: Good matches from ratio test

        Returns:
            Tuple of (homography matrix, inlier matches)
        """
        if len(matches) < self.config.min_matches:
            return None, []

        # Extract point correspondences
        src_pts = np.float32([[m.train_point.x, m.train_point.y] for m in matches]).reshape(-1, 1, 2)

        dst_pts = np.float32([[m.query_point.x, m.query_point.y] for m in matches]).reshape(-1, 1, 2)

        # Estimate homography using RANSAC
        try:
            homography, mask = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC, self.config.ransac_threshold, maxIters=2000, confidence=0.995
            )

            if homography is None:
                return None, []

            # Extract inlier matches
            inlier_matches = [matches[i] for i, is_inlier in enumerate(mask.ravel()) if is_inlier]

            return homography, inlier_matches

        except cv2.error as cv_error:
            if self.config.debug_mode:
                print(f"OpenCV error in homography estimation: {cv_error}")
            return None, []
        except Exception as exception:
            if self.config.debug_mode:
                print(f"Homography estimation failed: {exception}")
            return None, []

    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        if not self.matching_times:
            return {"avg_time_ms": 0, "fps": 0, "total_matches": 0}

        avg_time = np.mean(self.matching_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0

        return {
            "avg_time_ms": avg_time * 1000,
            "fps": fps,
            "total_matches": len(self.matching_times),
            "min_time_ms": min(self.matching_times) * 1000,
            "max_time_ms": max(self.matching_times) * 1000,
        }

    def clear_cache(self):
        """Clear cached clustering structures."""
        self.cached_clusters.clear()
        self.cached_descriptors.clear()


def compute_hough_matches(matches: List[Match]) -> List[Match]:
    """
    Apply Hough transform voting to filter matches by geometric consistency.

    Args:
        matches: Input matches

    Returns:
        Geometrically consistent matches
    """
    if len(matches) < 3:
        return matches

    # Create Hough accumulator
    hough_bins = HOUGH_BINS
    scale_bins = HOUGH_SCALE_BINS
    accumulator = np.zeros((hough_bins, scale_bins), dtype=np.int32)

    # Vote for each match
    match_votes = []
    for match in matches:
        key_pt = match.train_point
        query_pt = match.query_point

        # Compute scale and orientation
        scale = query_pt.scale / key_pt.scale if key_pt.scale > 0 else 1.0

        # Discretize scale
        scale_idx = min(int(scale * scale_bins / 4.0), scale_bins - 1)

        # Compute translation
        delta_x = query_pt.x - key_pt.x * scale
        delta_y = query_pt.y - key_pt.y * scale

        # Convert to angle
        angle = np.arctan2(delta_y, delta_x)
        angle_idx = int((angle + np.pi) * hough_bins / (2 * np.pi)) % hough_bins

        accumulator[angle_idx, scale_idx] += 1
        match_votes.append((angle_idx, scale_idx))

    # Find peak in accumulator
    max_votes = np.max(accumulator)
    if max_votes < 3:  # Minimum consensus
        return matches

    peak_indices = np.where(accumulator == max_votes)
    if len(peak_indices[0]) == 0:
        return matches

    peak_angle = peak_indices[0][0]
    peak_scale = peak_indices[1][0]

    # Filter matches that vote for the peak
    consistent_matches = []
    for i, (angle_idx, scale_idx) in enumerate(match_votes):
        # Allow some tolerance around peak
        angle_diff = min(abs(angle_idx - peak_angle), hough_bins - abs(angle_idx - peak_angle))
        scale_diff = abs(scale_idx - peak_scale)

        if angle_diff <= 2 and scale_diff <= 1:
            consistent_matches.append(matches[i])

    return consistent_matches if len(consistent_matches) >= 3 else matches
