# Moondream Real-time Object Detector

[![Tests](https://github.com/EdwardOwusuAdjei/moondream-vision/actions/workflows/tests.yml/badge.svg)](https://github.com/EdwardOwusuAdjei/moondream-vision/actions/workflows/tests.yml)
[![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen)](https://github.com/EdwardOwusuAdjei/moondream-vision)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This project uses the Moondream2 model to perform real-time object detection on a video stream from a webcam. It identifies specified objects and draws bounding boxes around them in the live video feed.

## Features

- **Real-time object detection** from a webcam feed or video files
- **Two operation modes**: Detection mode and Caption mode
  - **Detection mode**: Detects objects matching your prompt and labels them
  - **Caption mode**: Generates AI-powered detailed descriptions for detected objects
- **Bounding box visualization** with smart text wrapping
- **Configurable model and detection settings**
- **Well-structured and easy-to-understand codebase**

## Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/EdwardOwusuAdjei/moondream-vision.git
    cd moondream-vision
    ```

2.  **Create a virtual environment and activate it:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the application from the root directory of the project:

```bash
python main.py
```

This will start the webcam and begin detecting the default object ("what are the objects in the scene").

### Operation Modes

The application supports two modes:

#### Detection Mode (Default)
Detects objects matching your prompt and labels them with the prompt text:

```bash
python main.py -p "a person"
# or explicitly
python main.py -p "a person" -m detect
```

#### Caption Mode
Detects objects and generates detailed AI-powered captions for each detected region:

```bash
python main.py -p "a person" -m caption
```

In caption mode, the system will:
1. Detect objects matching your prompt
2. Crop each detected region
3. Generate a detailed caption (e.g., "a person wearing a blue shirt standing in a room")
4. Display the caption as the label

This is perfect for getting rich, contextual descriptions of what the AI sees!

### Custom Object Detection

You can specify different objects to detect using the `--prompt` or `-p` argument:

```bash
python main.py --prompt "a coffee mug"
python main.py -p "a pair of glasses"
python main.py -p "a laptop" -m caption  # With detailed captions
```

### Mac Users

If you encounter library issues on macOS, you may need to set the `DYLD_LIBRARY_PATH` environment variable:

```bash
DYLD_LIBRARY_PATH=$(brew --prefix vips)/lib python main.py
```

### Example Prompts and Detection Boxes

The system draws **green bounding boxes** around detected objects with labels. Here are some example prompts:

#### Person Detection
```bash
python main.py -p "a person"
```
- Draws a green box around the person's body/face
- Label: "a person"

#### Face Detection
```bash
python main.py -p "a person's face"
```
- Draws a box around the face region
- Label: "a person's face"

#### Object Detection
```bash
python main.py -p "a coffee mug"
python main.py -p "a pair of glasses"
python main.py -p "a laptop"
```
- Draws boxes around the specified objects
- Labels match the prompt text (in detection mode)

#### Caption Mode Examples
```bash
# Get detailed descriptions of detected people
python main.py -p "a person" -m caption

# Get detailed descriptions of objects
python main.py -p "a coffee mug" -m caption
python main.py -p "a laptop" -m caption
```
- Draws boxes around detected objects
- Labels are AI-generated captions with rich details
- Example captions: "a person wearing a blue shirt standing in a room", "a white coffee mug on a wooden table"

#### General Scene Query
```bash
python main.py  # Uses default prompt
```
- Detects the most prominent object in the scene
- Label: "what are the objects in the scene"

**Note:** Currently, the system detects up to 1 object per frame. Boxes are drawn as green rectangles with text labels that automatically wrap for readability.

Press `q` to quit the application.

## Testing

This project includes comprehensive tests with 98% code coverage. To run the tests:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src/moondream_realtime_detector --cov-report=html

# Run specific test file
pytest tests/unit/test_state.py
```

See [tests/README.md](tests/README.md) for more details on running and writing tests.

## Development

### Setting Up for Development

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Run tests: `pytest`

### Contributing

1. Write tests for new features
2. Ensure all tests pass: `pytest`
3. Maintain code coverage above 90%
4. Follow PEP 8 style guidelines
5. Update documentation as needed
