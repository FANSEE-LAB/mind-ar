"""
MindAR: High-performance real-time image recognition & tracking for AR applications.
"""

from .compiler import MindARCompiler
from .detector import Detector, FeaturePoint
from .matcher import Match, Matcher
from .tracker import Tracker

try:
    from importlib.metadata import version

    __version__ = version("mindar")
except ImportError:
    __version__ = "unknown"

__all__ = ["Detector", "Matcher", "Tracker", "MindARCompiler", "FeaturePoint", "Match"]
