# Test Suite

This directory contains comprehensive tests for the Moondream Real-time Object Detector project.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and pytest configuration
├── unit/                    # Unit tests
│   ├── test_state.py       # SharedState tests
│   ├── test_drawing.py      # Drawing utilities tests
│   └── test_detector.py    # Detector tests (with mocked model)
├── integration/            # Integration tests
│   └── test_video_processor.py  # VideoProcessor tests
└── fixtures/               # Test fixtures
```

## Running Tests

### Run All Tests

```bash
pytest
```

### Run with Coverage Report

```bash
pytest --cov=src/moondream_realtime_detector --cov-report=html
```

This will generate an HTML coverage report in `htmlcov/index.html`.

### Run Specific Test Categories

```bash
# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run specific test file
pytest tests/unit/test_state.py

# Run specific test
pytest tests/unit/test_state.py::TestSharedState::test_initialization
```

### Run with Verbose Output

```bash
pytest -v
```

### Run with Markers

```bash
# Run only unit tests (if marked)
pytest -m unit

# Run only integration tests (if marked)
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

## Test Coverage

The test suite covers:

- **SharedState**: Thread-safe state management
  - Initialization
  - Getter/Setter operations
  - Thread safety (concurrent reads/writes)
  - Edge cases (empty prompts, long prompts)

- **Drawing Utilities**: Bounding box and text rendering
  - Rectangle overlap detection
  - Bounding box drawing
  - Text wrapping and positioning
  - Info and prompt text rendering
  - Edge cases (empty objects, long text, boundary conditions)

- **Detector**: Model loading and inference
  - Model initialization
  - Object detection
  - Image captioning
  - Error handling
  - Configuration parameters

- **VideoProcessor**: Video processing pipeline
  - Initialization (webcam and file modes)
  - Frame processing
  - Detection and caption modes
  - Frame skipping logic
  - Resource cleanup
  - FPS calculation

## Test Fixtures

Common fixtures are defined in `conftest.py`:

- `sample_frame`: Sample video frame (numpy array)
- `sample_pil_image`: Sample PIL Image
- `sample_detected_objects`: Sample detection results
- `mock_model`: Mocked Moondream model
- `mock_video_capture`: Mocked OpenCV VideoCapture
- `mock_video_writer`: Mocked OpenCV VideoWriter

## Writing New Tests

When adding new tests:

1. Follow the existing test structure and naming conventions
2. Use descriptive test names: `test_<functionality>_<scenario>`
3. Include docstrings explaining what is being tested
4. Use fixtures from `conftest.py` when possible
5. Mock external dependencies (model, cv2, etc.)
6. Test both success and error cases
7. Test edge cases and boundary conditions

## Test Standards

- **AAA Pattern**: Arrange, Act, Assert
- **Isolation**: Each test should be independent
- **Clarity**: Tests should be easy to understand
- **Coverage**: Aim for >90% code coverage
- **Performance**: Tests should run quickly (use mocks)

## Continuous Integration

These tests are designed to run in CI/CD pipelines. The `pytest.ini` configuration includes:

- Coverage reporting
- Multiple output formats (terminal, HTML, XML)
- Strict marker validation
- Verbose output for better debugging

