"""
Feature point and match data structures.
Edge-optimized data types for feature detection and matching.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class FeaturePoint:
    """
    Represents a detected feature point with its descriptors.
    Optimized for memory efficiency on edge devices.

    Attributes:
        x (float): X coordinate in image
        y (float): Y coordinate in image
        scale (float): Scale at which feature was detected
        angle (float): Orientation angle in radians
        descriptors (List[int]): Binary descriptors
        maxima (bool): True if maxima, False if minima
        response (float): Feature response strength
        quality (float): Quality score [0-1]
    """

    x: float
    y: float
    scale: float
    angle: float
    descriptors: List[int]
    maxima: bool  # True for maxima, False if minima
    response: float
    quality: Optional[float] = None

    def __post_init__(self):
        if self.quality is None:
            self.quality = 0.5

    def get_position(self) -> Tuple[float, float]:
        """Get position as (x, y) tuple"""
        return (self.x, self.y)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "x": float(self.x),
            "y": float(self.y),
            "scale": float(self.scale),
            "angle": float(self.angle),
            "maxima": bool(self.maxima),
            "response": float(self.response),
            "quality": float(self.quality if self.quality is not None else 0.5),
            # Convert descriptors to integers for serialization
            "descriptors": [int(d) for d in self.descriptors[:8]],  # Store first 8 for efficiency
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeaturePoint":
        """Create from dictionary after deserialization"""
        return cls(
            x=data["x"],
            y=data["y"],
            scale=data["scale"],
            angle=data["angle"],
            maxima=data["maxima"],
            response=data.get("response", 1.0),
            descriptors=data.get("descriptors", []),
            quality=data.get("quality", 0.5),
        )


@dataclass
class Match:
    """
    Represents a match between two feature points.
    Optimized for memory efficiency on edge devices.

    Attributes:
        query_point (FeaturePoint): Feature point in query image
        key_point (FeaturePoint): Matching feature point in key image
        distance (float): Descriptor distance
        confidence (float): Match confidence [0-1]
    """

    query_point: FeaturePoint
    key_point: FeaturePoint
    distance: float
    confidence: Optional[float] = None

    def __post_init__(self):
        if self.confidence is None:
            # Convert distance to confidence (lower distance = higher confidence)
            self.confidence = max(0.0, 1.0 - self.distance / 100.0)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "query_point": self.query_point.to_dict(),
            "key_point": self.key_point.to_dict(),
            "distance": float(self.distance),
            "confidence": float(self.confidence if self.confidence is not None else 0.0),
        }


@dataclass
class DetectionResult:
    """
    Result of AR detection, optimized for edge devices.

    Attributes:
        target_id (int): Detected target ID
        homography (np.ndarray): 3x3 homography matrix
        matches (List[Match]): Feature matches
        inliers (int): Number of inlier matches
        confidence (float): Detection confidence [0-1]
        target_image (Optional[np.ndarray]): Target image if available
    """

    target_id: int
    homography: np.ndarray
    matches: List[Match]
    inliers: int
    confidence: float
    target_image: Optional[np.ndarray] = None

    def is_valid(self, min_inliers: int = 6, min_confidence: float = 0.5) -> bool:
        """Check if detection result is valid"""
        return self.inliers >= min_inliers and self.confidence >= min_confidence

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "target_id": int(self.target_id),
            "homography": self.homography.tolist(),
            "inliers": int(self.inliers),
            "confidence": float(self.confidence),
            "matches_count": len(self.matches),
            # Include only essential match data to save memory
            "matches": [m.to_dict() for m in self.matches[:10]],  # Store only first 10 matches
        }
