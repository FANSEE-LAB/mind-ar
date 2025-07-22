"""
MindAR Compiler utilities.

This module provides a MindAR-compatible compiler, allowing you to process target images and generate optimized binary files for efficient AR detection and matching.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import msgpack

from .detector import Detector


class MindARCompiler:
    """
    This compiler processes target images and creates optimized
    binary files for efficient AR detection, matching MindAR's format.
    """

    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode

    def compile_images(
        self, image_paths: List[Union[str, Path]], output_path: Union[str, Path], metadata: Optional[Dict] = None
    ) -> bool:
        """
        Compile multiple images into a single .mind file

        Args:
            image_paths: List of image file paths
            output_path: Output .mind file path
            metadata: Optional metadata for targets

        Returns:
            True if compilation successful
        """

        targets = []
        try:
            # Process each image
            for i, image_path in enumerate(image_paths):
                if self.debug_mode:
                    print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")

                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    print(f"Warning: Could not load {image_path}")
                    continue

                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Process the image and extract features
                target_data = self._process_single_image(gray, image, image_path)
                if target_data:
                    targets.append(target_data)

            # Create .mind file structure
            mind_data = {"v": 2, "dataList": targets}  # Version

            # Add metadata if provided
            if metadata:
                mind_data["metadata"] = metadata

            # Check if any targets were successfully processed
            if not targets:
                print(f"❌ No valid targets found in {len(image_paths)} images")
                return False

            # Serialize to MessagePack
            with open(output_path, "wb") as file:
                file.write(msgpack.packb(mind_data))

            print(f"✅ Compiled {len(targets)} targets to {output_path}")
            return True

        except (IOError, OSError) as io_error:
            print(f"❌ File I/O error during compilation: {io_error}")
            return False
        except ValueError as value_error:
            print(f"❌ Data processing error during compilation: {value_error}")
            return False
        except Exception as exception:
            print(f"❌ Compilation failed: {exception}")
            return False

    def _process_single_image(self, gray, image, image_path) -> Optional[Dict]:
        """Process a single image and extract features."""
        try:
            from .detector import DetectorConfig

            config = DetectorConfig(method="hybrid", debug_mode=self.debug_mode)
            detector = Detector(config)
            result = detector.detect(gray)
            feature_points = result["feature_points"]

            maxima_points = []
            minima_points = []

            for feature_point in feature_points:
                point_data = {
                    "x": feature_point.x,
                    "y": feature_point.y,
                    "scale": feature_point.scale,
                    "angle": feature_point.angle,
                    "descriptors": feature_point.descriptors,
                }

                if feature_point.maxima:
                    maxima_points.append(point_data)
                else:
                    minima_points.append(point_data)

            # Create target data (store all feature points for matching)
            target_data = {
                "width": image.shape[1],
                "height": image.shape[0],
                "scale": 1.0,
                "featurePoints": maxima_points + minima_points,
            }

            if self.debug_mode:
                print(
                    f"  - Extracted {len(maxima_points) + len(minima_points)} features "
                    f"({len(maxima_points)} maxima, {len(minima_points)} minima)"
                )

            return target_data

        except RuntimeError as runtime_error:
            print(f"⚠️ Feature extraction failed for {image_path}: {runtime_error}")
            return None
        except Exception as exception:
            print(f"⚠️ Unexpected error processing {image_path}: {exception}")
            return None

    def _build_hierarchical_cluster(self, points: List[Dict]) -> Dict:
        """Build hierarchical clustering for points"""
        if len(points) == 0:
            return {"rootNode": {"leaf": True, "pointIndexes": []}}

        # Simple hierarchical clustering
        # In a real implementation, this would use k-means or similar
        root_node = {"leaf": False, "children": []}

        # For simplicity, create a single cluster with all points
        child_node = {"leaf": True, "centerPointIndex": 0, "pointIndexes": list(range(len(points)))}

        root_node["children"].append(child_node)

        return {"rootNode": root_node}

    def compile_directory(
        self, input_dir: Union[str, Path], output_path: Union[str, Path], image_extensions: List[str] = None
    ) -> bool:
        """
        Compile all images in a directory

        Args:
            input_dir: Directory containing images
            output_path: Output .mind file path
            image_extensions: List of image extensions to include

        Returns:
            True if compilation successful
        """
        if image_extensions is None:
            image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"❌ Input directory not found: {input_dir}")
            return False

        # Find all image files
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(input_path.glob(f"*{ext}"))
            image_paths.extend(input_path.glob(f"*{ext.upper()}"))

        if not image_paths:
            print(f"❌ No image files found in {input_dir}")
            return False

        # Load metadata if available
        metadata = None
        metadata_file = input_path / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r", encoding="utf-8") as file:
                    metadata = json.load(file)
                print(f"📋 Loaded metadata from {metadata_file}")
            except (IOError, json.JSONDecodeError) as json_error:
                print(f"⚠️ Could not load metadata: {json_error}")

        return self.compile_images(image_paths, output_path, metadata)

    def load_mind_file(self, mind_path: Union[str, Path]) -> Optional[Dict]:
        """
        Load and parse a .mind file

        Args:
            mind_path: Path to .mind file

        Returns:
            Parsed mind data or None if failed
        """

        try:
            with open(mind_path, "rb") as file:
                data = msgpack.unpackb(file.read(), raw=False)

            if data.get("v") != 2:
                print(f"⚠️ Unsupported .mind file version: {data.get('v')}")
                return None

            return data

        except (IOError, OSError) as io_error:
            print(f"❌ Failed to read .mind file: {io_error}")
            return None
        except (
            msgpack.exceptions.ExtraData,
            msgpack.exceptions.UnpackException,
            msgpack.exceptions.UnpackValueError,
        ) as msgpack_error:
            print(f"❌ Failed to parse .mind file: {msgpack_error}")
            return None
        except Exception as exception:
            print(f"❌ Unexpected error loading .mind file: {exception}")
            return None
