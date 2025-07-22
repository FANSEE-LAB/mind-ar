"""
Pytest configuration and fixtures for MindAR tests.
Provides test data and utilities for cross-version compatibility testing.
"""

import os
import sys
from pathlib import Path
from typing import Generator, List

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def test_images_dir() -> Path:
    """Create test images directory with sample images."""
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(exist_ok=True)

    # Create simple test images if they don't exist
    images_dir = test_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # Create a simple test pattern image
    test_image_path = images_dir / "test_pattern.png"
    if not test_image_path.exists():
        import cv2

        # Create a simple checkerboard pattern
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[::20, ::20] = 255  # Create checkerboard pattern
        cv2.imwrite(str(test_image_path), img)

    return images_dir


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample test image with features."""
    # Create a simple image with corners and edges
    img = np.zeros((200, 200), dtype=np.uint8)

    # Add some geometric shapes for feature detection
    # Rectangle
    img[50:150, 50:150] = 255
    # Cross pattern
    img[90:110, :] = 255
    img[:, 90:110] = 255

    return img


@pytest.fixture
def sample_color_image() -> np.ndarray:
    """Create a sample color test image."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)

    # Add colored rectangles
    img[50:100, 50:100] = [255, 0, 0]  # Red
    img[100:150, 100:150] = [0, 255, 0]  # Green
    img[75:125, 75:125] = [0, 0, 255]  # Blue

    return img


@pytest.fixture
def feature_points() -> List:
    """Create sample feature points for testing."""
    from mindar.types import FeaturePoint

    points = []
    for i in range(10):
        point = FeaturePoint(
            x=float(i * 20),
            y=float(i * 15),
            scale=1.0 + i * 0.1,
            angle=i * 0.5,
            descriptors=[i % 256] * 64,  # 64-byte descriptor
            maxima=i % 2 == 0,
            response=0.5 + i * 0.1,
            quality=0.6 + i * 0.05,
        )
        points.append(point)

    return points


@pytest.fixture
def matches() -> List:
    """Create sample matches for testing."""
    from mindar.types import FeaturePoint, Match

    matches = []
    for i in range(5):
        query_point = FeaturePoint(
            x=float(i * 10),
            y=float(i * 10),
            scale=1.0,
            angle=0.0,
            descriptors=[i] * 64,
            maxima=True,
            response=0.5,
            quality=0.7,
        )

        key_point = FeaturePoint(
            x=float(i * 10 + 2),
            y=float(i * 10 + 1),
            scale=1.0,
            angle=0.1,
            descriptors=[i + 1] * 64,
            maxima=True,
            response=0.6,
            quality=0.8,
        )

        match = Match(query_point=query_point, key_point=key_point, distance=float(i * 5), confidence=0.8 - i * 0.1)
        matches.append(match)

    return matches


@pytest.fixture
def python_versions() -> List[str]:
    """Get list of Python versions to test compatibility."""
    return ["3.9", "3.10", "3.11", "3.12"]


@pytest.fixture
def numpy_versions() -> List[str]:
    """Get list of numpy versions to test compatibility."""
    return ["1.24.0", "1.26.0", "1.28.0"]


@pytest.fixture
def opencv_versions() -> List[str]:
    """Get list of opencv versions to test compatibility."""
    return ["4.8.0", "4.9.0", "4.10.0"]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "compatibility: marks tests for version compatibility")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark tests with "performance" in name as performance tests
        if "performance" in item.name.lower():
            item.add_marker(pytest.mark.performance)

        # Mark tests with "integration" in name as integration tests
        if "integration" in item.name.lower():
            item.add_marker(pytest.mark.integration)

        # Mark tests with "compatibility" in name as compatibility tests
        if "compatibility" in item.name.lower():
            item.add_marker(pytest.mark.compatibility)
