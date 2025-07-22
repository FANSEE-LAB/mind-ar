# MindAR Test Suite

This directory contains comprehensive tests for the MindAR package, ensuring compatibility across Python versions and thorough testing of core AR functionality.

## Test Structure

### Core Test Files

- **`test_types.py`** - Tests for data structures (FeaturePoint, Match, DetectionResult)
- **`test_detector.py`** - Tests for feature detection algorithms
- **`test_matcher.py`** - Tests for feature matching algorithms
- **`test_tracker.py`** - Tests for object tracking functionality
- **`test_compiler.py`** - Tests for image compilation and .mind file generation
- **`test_integration.py`** - End-to-end pipeline tests
- **`test_compatibility.py`** - Cross-version compatibility tests
- **`test_performance.py`** - Performance benchmarks and requirements

### Configuration

- **`conftest.py`** - Pytest configuration and shared fixtures
- **`pytest.ini`** - Pytest settings and markers

## Test Categories

### Unit Tests

- Individual component testing
- Data structure validation
- Algorithm correctness

### Integration Tests

- Complete pipeline testing
- Real-world scenario simulation
- End-to-end workflows

### Performance Tests

- Speed benchmarks
- Memory usage monitoring
- Scalability testing

### Compatibility Tests

- Python version compatibility (3.9+)
- Dependency version testing
- Platform compatibility

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mindar

# Run specific test file
pytest tests/test_detector.py
```

### Test Categories

```bash
# Run only unit tests (exclude slow/integration/performance)
pytest -m "not slow and not integration and not performance"

# Run only integration tests
pytest -m integration

# Run only performance tests
pytest -m performance

# Run only compatibility tests
pytest -m compatibility
```

### Performance Testing

```bash
# Run performance tests with timing
pytest -m performance -v

# Run with detailed timing
pytest -m performance --durations=10
```

### Cross-Version Testing

```bash
# Test with different Python versions
python3.9 -m pytest tests/test_compatibility.py
python3.10 -m pytest tests/test_compatibility.py
python3.11 -m pytest tests/test_compatibility.py
python3.12 -m pytest tests/test_compatibility.py
```

## Test Requirements

### Dependencies

- pytest >= 8.0.0
- pytest-cov >= 6.0.0
- numpy >= 1.24.0
- opencv-python >= 4.8.0
- msgpack >= 1.0.0
- psutil (for performance tests)

### Optional Dependencies

- numba >= 0.59.0 (for JIT acceleration tests)

## Test Data

Test images and data are generated automatically by fixtures in `conftest.py`:

- Sample images with geometric patterns
- Feature points with various characteristics
- Match data for testing algorithms

## Coverage Requirements

- **Minimum coverage**: 90%
- **Critical modules**: 95% (detector, matcher, tracker)
- **Data structures**: 100%

## Performance Benchmarks

### Detection Performance

- Average time: < 100ms per image
- Maximum time: < 200ms per image
- Memory usage: < 50MB increase

### Matching Performance

- Average time: < 50ms per match
- Maximum time: < 100ms per match
- Cache efficiency: > 2x speedup

### Tracking Performance

- Average time: < 50ms per frame
- Maximum time: < 100ms per frame
- Real-time capable: 30 FPS

### Compilation Performance

- Average time: < 1 second per image
- Memory usage: < 100MB increase
- Scalable to 100+ images

## Continuous Integration

Tests are designed to run in CI environments:

- GitHub Actions
- GitLab CI
- Jenkins
- Local development

## Debugging Tests

### Verbose Output

```bash
pytest -v -s
```

### Debug Specific Test

```bash
pytest tests/test_detector.py::TestDetector::test_detect_fast_method -v -s
```

### Coverage Report

```bash
pytest --cov=mindar --cov-report=html
# Open htmlcov/index.html in browser
```

## Test Maintenance

### Adding New Tests

1. Follow naming convention: `test_*.py`
2. Use appropriate markers
3. Include docstrings for test methods
4. Add to relevant test category

### Updating Tests

1. Maintain backward compatibility
2. Update performance benchmarks as needed
3. Ensure cross-version compatibility
4. Update documentation

### Test Data Management

1. Use fixtures for reusable test data
2. Clean up temporary files
3. Use appropriate test image sizes
4. Include edge cases and error conditions
