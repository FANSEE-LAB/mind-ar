"""
ARtify-Core: High-performance real-time image recognition & tracking.

This package provides optimized feature detection, matching, and tracking
for AR applications on edge devices with minimal dependencies.
"""

from .compiler import MindARCompiler
from .detector import Detector, FeaturePoint
from .matcher import Match, Matcher
from .tracker import Tracker

__version__ = "0.2.0"
__author__ = "FANSEE <info@fansee.studio>"
__license__ = "MIT"

__all__ = ["Detector", "Matcher", "Tracker", "MindARCompiler", "FeaturePoint", "Match"]
