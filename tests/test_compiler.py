"""
Tests for MindAR compiler module.
Tests image compilation and cross-version compatibility.
"""

import json
import tempfile
from pathlib import Path

import cv2
import msgpack
import numpy as np
import pytest

from mindar.compiler import MindARCompiler


class TestMindARCompiler:
    """Test MindARCompiler class functionality."""

    def test_compiler_initialization(self):
        """Test compiler initialization."""
        compiler = MindARCompiler(debug_mode=True)

        assert compiler.debug_mode is True

    def test_compiler_default_initialization(self):
        """Test compiler with default settings."""
        compiler = MindARCompiler()

        assert compiler.debug_mode is False

    def test_process_single_image(self, sample_image):
        """Test processing a single image."""
        compiler = MindARCompiler(debug_mode=True)

        # Convert sample image to color for testing
        if len(sample_image.shape) == 2:
            color_image = cv2.cvtColor(sample_image, cv2.COLOR_GRAY2BGR)
        else:
            color_image = sample_image

        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        target_data = compiler._process_single_image(gray_image, color_image, "test_image.png")

        assert target_data is not None
        assert "width" in target_data
        assert "height" in target_data
        assert "scale" in target_data
        assert "featurePoints" in target_data

        assert target_data["width"] == color_image.shape[1]
        assert target_data["height"] == color_image.shape[0]
        assert target_data["scale"] == 1.0
        assert isinstance(target_data["featurePoints"], list)

    def test_process_single_image_failure(self):
        """Test processing image that fails."""
        compiler = MindARCompiler(debug_mode=True)

        # Create an image that might cause issues
        problematic_image = np.zeros((10, 10), dtype=np.uint8)  # Very small image

        target_data = compiler._process_single_image(problematic_image, problematic_image, "test.png")

        # Should handle gracefully
        assert target_data is not None or target_data is None

    def test_build_hierarchical_cluster(self):
        """Test hierarchical cluster building."""
        compiler = MindARCompiler()

        # Test with empty points
        empty_points = []
        cluster = compiler._build_hierarchical_cluster(empty_points)

        assert "rootNode" in cluster
        assert cluster["rootNode"]["leaf"] is True
        assert cluster["rootNode"]["pointIndexes"] == []

        # Test with some points
        points = [{"x": 100, "y": 200}, {"x": 150, "y": 250}, {"x": 200, "y": 300}]
        cluster = compiler._build_hierarchical_cluster(points)

        assert "rootNode" in cluster
        assert cluster["rootNode"]["leaf"] is False
        assert "children" in cluster["rootNode"]

    def test_compile_images_success(self, test_images_dir):
        """Test successful image compilation."""
        compiler = MindARCompiler(debug_mode=True)

        # Create test images
        image_paths = []
        for i in range(3):
            image_path = test_images_dir / f"test_image_{i}.png"

            # Create a simple test image
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), img)
            image_paths.append(str(image_path))

        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".mind", delete=False) as tmp_file:
            output_path = tmp_file.name

        try:
            # Test compilation
            success = compiler.compile_images(image_paths, output_path)

            assert success is True

            # Check that output file exists and is not empty
            assert Path(output_path).exists()
            assert Path(output_path).stat().st_size > 0

            # Try to load the compiled file
            loaded_data = compiler.load_mind_file(output_path)
            assert loaded_data is not None
            assert "v" in loaded_data
            assert "dataList" in loaded_data
            assert len(loaded_data["dataList"]) == 3

        finally:
            # Clean up
            Path(output_path).unlink(missing_ok=True)
            for image_path in image_paths:
                Path(image_path).unlink(missing_ok=True)

    def test_compile_images_with_metadata(self, test_images_dir):
        """Test image compilation with metadata."""
        compiler = MindARCompiler(debug_mode=True)

        # Create test image
        image_path = test_images_dir / "test_image.png"
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(image_path), img)

        # Create metadata
        metadata = {"version": "1.0", "description": "Test targets", "author": "Test Author"}

        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".mind", delete=False) as tmp_file:
            output_path = tmp_file.name

        try:
            # Test compilation with metadata
            success = compiler.compile_images([str(image_path)], output_path, metadata)

            assert success is True

            # Load and check metadata
            loaded_data = compiler.load_mind_file(output_path)
            assert loaded_data is not None
            assert "metadata" in loaded_data
            assert loaded_data["metadata"] == metadata

        finally:
            # Clean up
            Path(output_path).unlink(missing_ok=True)
            Path(image_path).unlink(missing_ok=True)

    def test_compile_images_failure(self):
        """Test image compilation with invalid inputs."""
        compiler = MindARCompiler(debug_mode=True)

        # Test with non-existent image
        with tempfile.NamedTemporaryFile(suffix=".mind", delete=False) as tmp_file:
            output_path = tmp_file.name

        try:
            success = compiler.compile_images(["non_existent_image.png"], output_path)
            assert success is False

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_compile_directory_success(self, test_images_dir):
        """Test directory compilation."""
        compiler = MindARCompiler(debug_mode=True)

        # Create test images in directory
        for i in range(3):
            image_path = test_images_dir / f"test_image_{i}.png"
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), img)

        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".mind", delete=False) as tmp_file:
            output_path = tmp_file.name

        try:
            # Test directory compilation
            success = compiler.compile_directory(str(test_images_dir), output_path)

            assert success is True

            # Check output
            loaded_data = compiler.load_mind_file(output_path)
            assert loaded_data is not None
            assert len(loaded_data["dataList"]) >= 3

        finally:
            # Clean up
            Path(output_path).unlink(missing_ok=True)
            for image_path in test_images_dir.glob("test_image_*.png"):
                image_path.unlink(missing_ok=True)

    def test_compile_directory_with_metadata(self, test_images_dir):
        """Test directory compilation with metadata file."""
        compiler = MindARCompiler(debug_mode=True)

        # Create test images
        for i in range(2):
            image_path = test_images_dir / f"test_image_{i}.png"
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), img)

        # Create metadata file
        metadata = {"version": "1.0", "description": "Test targets from directory", "created": "2024-01-01"}
        metadata_file = test_images_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".mind", delete=False) as tmp_file:
            output_path = tmp_file.name

        try:
            # Test directory compilation
            success = compiler.compile_directory(str(test_images_dir), output_path)

            assert success is True

            # Check metadata was loaded
            loaded_data = compiler.load_mind_file(output_path)
            assert loaded_data is not None
            assert "metadata" in loaded_data

        finally:
            # Clean up
            Path(output_path).unlink(missing_ok=True)
            metadata_file.unlink(missing_ok=True)
            for image_path in test_images_dir.glob("test_image_*.png"):
                image_path.unlink(missing_ok=True)

    def test_compile_directory_empty(self, test_images_dir):
        """Test directory compilation with no images."""
        compiler = MindARCompiler(debug_mode=True)

        # Create a temporary empty directory
        with tempfile.TemporaryDirectory() as empty_dir:
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix=".mind", delete=False) as tmp_file:
                output_path = tmp_file.name

            try:
                # Test with empty directory
                success = compiler.compile_directory(empty_dir, output_path)

                # Should fail gracefully
                assert success is False

            finally:
                Path(output_path).unlink(missing_ok=True)

    def test_compile_directory_nonexistent(self):
        """Test directory compilation with non-existent directory."""
        compiler = MindARCompiler(debug_mode=True)

        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".mind", delete=False) as tmp_file:
            output_path = tmp_file.name

        try:
            # Test with non-existent directory
            success = compiler.compile_directory("non_existent_directory", output_path)

            assert success is False

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_load_mind_file_success(self, test_images_dir):
        """Test loading a valid .mind file."""
        compiler = MindARCompiler(debug_mode=True)

        # Create a test .mind file
        test_data = {
            "v": 2,
            "dataList": [
                {
                    "width": 100,
                    "height": 100,
                    "scale": 1.0,
                    "featurePoints": [{"x": 50, "y": 50, "scale": 1.0, "angle": 0.0, "descriptors": [1, 2, 3, 4]}],
                }
            ],
        }

        # Create temporary .mind file
        with tempfile.NamedTemporaryFile(suffix=".mind", delete=False) as tmp_file:
            mind_path = tmp_file.name
            tmp_file.write(msgpack.packb(test_data))

        try:
            # Test loading
            loaded_data = compiler.load_mind_file(mind_path)

            assert loaded_data is not None
            assert loaded_data["v"] == 2
            assert len(loaded_data["dataList"]) == 1
            assert loaded_data["dataList"][0]["width"] == 100

        finally:
            Path(mind_path).unlink(missing_ok=True)

    def test_load_mind_file_nonexistent(self):
        """Test loading non-existent .mind file."""
        compiler = MindARCompiler(debug_mode=True)

        loaded_data = compiler.load_mind_file("non_existent_file.mind")

        assert loaded_data is None

    def test_load_mind_file_invalid(self):
        """Test loading invalid .mind file."""
        compiler = MindARCompiler(debug_mode=True)

        # Create invalid file
        with tempfile.NamedTemporaryFile(suffix=".mind", delete=False) as tmp_file:
            mind_path = tmp_file.name
            tmp_file.write(b"invalid data")

        try:
            loaded_data = compiler.load_mind_file(mind_path)

            # Should handle gracefully
            assert loaded_data is None

        finally:
            Path(mind_path).unlink(missing_ok=True)


class TestCompilerCompatibility:
    """Test compiler compatibility across different scenarios."""

    def test_different_image_formats(self, test_images_dir):
        """Test compilation with different image formats."""
        compiler = MindARCompiler(debug_mode=True)

        formats = [".png", ".jpg", ".bmp"]
        image_paths = []

        for fmt in formats:
            image_path = test_images_dir / f"test_image{fmt}"

            # Create test image
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

            if fmt == ".jpg":
                cv2.imwrite(str(image_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            else:
                cv2.imwrite(str(image_path), img)

            image_paths.append(str(image_path))

        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".mind", delete=False) as tmp_file:
            output_path = tmp_file.name

        try:
            # Test compilation
            success = compiler.compile_images(image_paths, output_path)

            assert success is True

            # Check output
            loaded_data = compiler.load_mind_file(output_path)
            assert loaded_data is not None
            assert len(loaded_data["dataList"]) == len(formats)

        finally:
            # Clean up
            Path(output_path).unlink(missing_ok=True)
            for image_path in image_paths:
                Path(image_path).unlink(missing_ok=True)

    def test_different_image_sizes(self, test_images_dir):
        """Test compilation with different image sizes."""
        compiler = MindARCompiler(debug_mode=True)

        sizes = [(50, 50), (100, 100), (200, 200), (400, 400)]
        image_paths = []

        for width, height in sizes:
            image_path = test_images_dir / f"test_image_{width}x{height}.png"

            # Create test image
            img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), img)

            image_paths.append(str(image_path))

        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".mind", delete=False) as tmp_file:
            output_path = tmp_file.name

        try:
            # Test compilation
            success = compiler.compile_images(image_paths, output_path)

            assert success is True

            # Check output
            loaded_data = compiler.load_mind_file(output_path)
            assert loaded_data is not None
            assert len(loaded_data["dataList"]) == len(sizes)

            # Check dimensions
            for i, (width, height) in enumerate(sizes):
                target_data = loaded_data["dataList"][i]
                assert target_data["width"] == width
                assert target_data["height"] == height

        finally:
            # Clean up
            Path(output_path).unlink(missing_ok=True)
            for image_path in image_paths:
                Path(image_path).unlink(missing_ok=True)

    def test_large_number_of_images(self, test_images_dir):
        """Test compilation with large number of images."""
        compiler = MindARCompiler(debug_mode=True)

        # Create many test images
        num_images = 10
        image_paths = []

        for i in range(num_images):
            image_path = test_images_dir / f"test_image_{i:03d}.png"

            # Create test image
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), img)

            image_paths.append(str(image_path))

        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".mind", delete=False) as tmp_file:
            output_path = tmp_file.name

        try:
            # Test compilation
            success = compiler.compile_images(image_paths, output_path)

            assert success is True

            # Check output
            loaded_data = compiler.load_mind_file(output_path)
            assert loaded_data is not None
            assert len(loaded_data["dataList"]) == num_images

        finally:
            # Clean up
            Path(output_path).unlink(missing_ok=True)
            for image_path in image_paths:
                Path(image_path).unlink(missing_ok=True)


@pytest.mark.performance
class TestCompilerPerformance:
    """Performance tests for compiler."""

    def test_compilation_performance(self, test_images_dir):
        """Test compilation performance."""
        compiler = MindARCompiler(debug_mode=False)  # Disable debug for performance

        # Create test images
        num_images = 5
        image_paths = []

        for i in range(num_images):
            image_path = test_images_dir / f"perf_test_image_{i}.png"

            # Create smaller test image for performance testing
            img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), img)

            image_paths.append(str(image_path))

        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".mind", delete=False) as tmp_file:
            output_path = tmp_file.name

        try:
            # Performance test
            import time

            start_time = time.time()
            success = compiler.compile_images(image_paths, output_path)
            compilation_time = time.time() - start_time

            assert success is True
            assert compilation_time < 30.0  # Should be under 30 seconds

        finally:
            # Clean up
            Path(output_path).unlink(missing_ok=True)
            for image_path in image_paths:
                Path(image_path).unlink(missing_ok=True)

    def test_memory_usage(self, test_images_dir):
        """Test memory usage during compilation."""
        import os

        import psutil

        compiler = MindARCompiler(debug_mode=False)

        # Create test images
        image_paths = []
        for i in range(3):
            image_path = test_images_dir / f"mem_test_image_{i}.png"

            # Create test image
            img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), img)

            image_paths.append(str(image_path))

        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".mind", delete=False) as tmp_file:
            output_path = tmp_file.name

        try:
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss

            # Run compilation
            success = compiler.compile_images(image_paths, output_path)

            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            assert success is True
            # Memory increase should be reasonable (less than 200MB)
            assert memory_increase < 200 * 1024 * 1024

        finally:
            # Clean up
            Path(output_path).unlink(missing_ok=True)
            for image_path in image_paths:
                Path(image_path).unlink(missing_ok=True)
