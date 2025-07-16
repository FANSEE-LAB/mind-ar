# MindAR Python

High-performance MindAR implementation for real-time image recognition on edge devices.

## Features

- **MindAR Compatible**: Full compatibility with MindAR .mind file format
- **High Performance**: Optimized with numba JIT compilation for edge devices
- **Real-time Detection**: Efficient feature detection and matching
- **Modern Architecture**: Clean configuration-based API with dataclasses
- **Type Safe**: Full type hints and proper error handling
- **Production Ready**: Comprehensive testing and linting (pylint score 7.5+)

## Installation

### From PyPI (Recommended)

```bash
pip install mindar
```

### From Source

```bash
git clone https://github.com/FANSEE-LAB/mind-ar.git
cd mind-ar
pip install -e .
```

### Requirements

- Python >= 3.9 (required for numba optimization)
- OpenCV
- NumPy
- msgpack (for .mind file format)
- numba (for performance optimization)

## Usage

### Basic Detection

```python
import cv2
from mindar import Detector, Matcher, MindARCompiler
from mindar.types import DetectorConfig, MatcherConfig

# Configure detector with new configuration system
detector_config = DetectorConfig(
    method="super_hybrid",
    max_features=1000,
    debug_mode=False
)
detector = Detector(detector_config)

# Configure matcher
matcher_config = MatcherConfig(
    ratio_threshold=0.75,
    min_matches=8,
    debug_mode=False
)
matcher = Matcher(matcher_config)

# Detect features in image
image = cv2.imread("target.jpg", cv2.IMREAD_GRAYSCALE)
result = detector.detect(image)
feature_points = result["feature_points"]

print(f"Detected {len(feature_points)} features")
```

### Compile .mind Files

```python
from mindar.compiler import MindARCompiler

# Initialize compiler with debug mode
compiler = MindARCompiler(debug_mode=True)

# Compile images to .mind file
success = compiler.compile_directory("./images", "./targets.mind")
if success:
    print("✅ Compilation successful")

# Load compiled targets
mind_data = compiler.load_mind_file("./targets.mind")
print(f"Loaded {len(mind_data['dataList'])} targets")
```

## Performance

Optimized for edge devices like Raspberry Pi 4:

- **Detection**: ~50ms per frame (640x480)
- **Matching**: ~20ms per target
- **Memory**: <100MB usage

## License

MIT License - Compatible with original MindAR project
