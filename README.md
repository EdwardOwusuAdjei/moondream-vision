# Moondream Real-time Object Detector

[![Tests](https://github.com/EdwardOwusuAdjei/moondream-vision/actions/workflows/tests.yml/badge.svg)](https://github.com/EdwardOwusuAdjei/moondream-vision/actions/workflows/tests.yml)
[![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen)](https://github.com/EdwardOwusuAdjei/moondream-vision)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This project uses the Moondream2 model to perform real-time object detection on a video stream from a webcam. It identifies specified objects and draws bounding boxes around them in the live video feed.

## Example Output

See the detection and captioning in action! Watch the demo below:

![Demo GIF](demo/demo.gif)



## Features

- **Real-time object detection** from a webcam feed or video files
- **Interactive prompt updates** - Change detection prompts on the fly without restarting
- **Two operation modes**: Detection mode and Caption mode
  - **Detection mode**: Detects objects matching your prompt and labels them
  - **Caption mode**: Generates AI-powered detailed descriptions for detected objects
- **Bounding box visualization** with smart text wrapping
- **Video output support** - Save processed videos with annotations
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

4.  **Test the installation (optional):**

    The repository includes test videos in the `videos/` folder. You can test the installation with:

    ```bash
    python main.py -f videos/Gus.mp4 -p "what is going on?" -m caption
    ```

## Usage

### Webcam Input (Default)

Run the application from the root directory of the project:

```bash
python main.py
```

This will start the webcam and begin detecting the default object ("what are the objects in the scene").

### Interactive Prompt Updates

While the application is running, you can **update the detection prompt interactively** without restarting:

1. The application will display: `Press Enter to change the prompt. Type 'quit' to exit.`
2. Type a new prompt and press Enter to update detection in real-time
3. Type `quit` to exit the application

**Example:**
```bash
$ python main.py -p "a person"
Starting Moondream real-time object detector...
Initial prompt: 'a person'
Running in mode: 'detect'

Press Enter to change the prompt. Type 'quit' to exit.
New prompt: a coffee mug
Prompt updated to: 'a coffee mug'
New prompt: a laptop
Prompt updated to: 'a laptop'
New prompt: quit
```

This allows you to experiment with different prompts on the fly without stopping the video feed!

### Video File Input

You can also process video files instead of using the webcam. Video files should be placed in the `videos/` folder:

```bash
# Process a video file from videos/ directory
python main.py -f videos/Gus.mp4

# Process a video with a custom prompt
python main.py -f videos/Gus.mp4 -p "a person"

# Process a video in caption mode (recommended for testing)
python main.py -f videos/Gus.mp4 -p "what is going on?" -m caption

# Process a video and save the output
python main.py -f videos/Gus.mp4 -o output.mp4 -p "a person"

# Full example: Process video, generate captions, and save to output
python main.py -f videos/Gus.mp4 -o output.mp4 -p "what is going on?" -m caption
```

**Testing with video files:**

The repository includes test videos in the `videos/` folder (e.g., `videos/Gus.mp4`):

```bash
# Test with videos from videos/ directory
python main.py -f videos/Gus.mp4 -p "what is going on?" -m caption

# Test with detection mode
python main.py -f videos/Gus.mp4 -p "a person"

# Test and save output
python main.py -f videos/Gus.mp4 -o output.mp4 -p "a person" -m caption
```

**Supported video formats:** MP4, AVI, MOV, MKV, and other formats supported by OpenCV.

**Note:** When processing video files, the application will respect the original video's frame rate for playback.

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

**Tip:** You can also update prompts interactively while the application is running (see Interactive Prompt Updates above).

### Mac Users

If you encounter library issues on macOS, you may need to set the `DYLD_LIBRARY_PATH` environment variable:

```bash
DYLD_LIBRARY_PATH=$(brew --prefix vips)/lib python main.py
```

## Examples

### Quick Start Examples

#### 1. Webcam with Default Settings
```bash
# Start webcam with default prompt
python main.py
```

#### 2. Webcam with Custom Prompt
```bash
# Detect a person in webcam feed
python main.py -p "a person"

# Detect faces
python main.py -p "a person's face"

# Detect objects
python main.py -p "a coffee mug"
```

#### 3. Video File Processing (Recommended for Testing)
```bash
# Process video from videos/ directory with caption mode
python main.py -f videos/Gus.mp4 -p "what is going on?" -m caption

# Process and save output
python main.py -f videos/Gus.mp4 -o output.mp4 -p "a person" -m caption
```

#### 4. Mac Users
```bash
# With DYLD_LIBRARY_PATH for macOS (using videos/ directory)
DYLD_LIBRARY_PATH=$(brew --prefix vips)/lib python main.py -f videos/Gus.mp4 -p "what is going on?" -m caption

# Save output on Mac
DYLD_LIBRARY_PATH=$(brew --prefix vips)/lib python main.py -f videos/Gus.mp4 -o output.mp4 -p "a person" -m caption
```

### Complete Usage Examples

#### Example 1: Person Detection (Webcam)
```bash
python main.py -p "a person"
```
- **Input:** Webcam feed
- **Mode:** Detection
- **Result:** Green boxes around detected people with "a person" label

#### Example 2: Face Detection (Webcam)
```bash
python main.py -p "a person's face"
```
- **Input:** Webcam feed
- **Mode:** Detection
- **Result:** Boxes around faces with "a person's face" label

#### Example 3: Object Detection with Caption (Video File)
```bash
python main.py -f videos/Gus.mp4 -p "a person" -m caption
```
- **Input:** Video file (videos/Gus.mp4)
- **Mode:** Caption
- **Result:** Boxes with detailed captions like "a person wearing a blue shirt standing in a room"

#### Example 4: Scene Analysis (Video File)
```bash
python main.py -f videos/Gus.mp4 -p "what is going on?" -m caption
```
- **Input:** Video file (videos/Gus.mp4)
- **Mode:** Caption
- **Result:** Detailed scene descriptions for detected objects

#### Example 5: Process and Save Video
```bash
python main.py -f videos/Gus.mp4 -o output.mp4 -p "a person" -m caption
```
- **Input:** videos/Gus.mp4
- **Output:** output.mp4 (saved with annotations)
- **Mode:** Caption
- **Result:** Processed video saved to disk

#### Example 6: Multiple Object Types
```bash
# Detect coffee mugs
python main.py -f videos/Gus.mp4 -p "a coffee mug" -m caption

# Detect laptops
python main.py -f videos/Gus.mp4 -p "a laptop" -m caption

# Detect glasses
python main.py -f videos/Gus.mp4 -p "a pair of glasses" -m caption
```

### Interactive Examples

#### Example 7: Interactive Prompt Updates
```bash
# Start with one prompt, then change it interactively
python main.py -p "a person"

# While running, type in terminal:
# New prompt: a coffee mug
# New prompt: a laptop
# New prompt: quit
```

### Example Prompts and Detection Boxes

The system draws **green bounding boxes** around detected objects with labels. Here are example prompts:

#### Person Detection
```bash
python main.py -p "a person"
python main.py -f videos/Gus.mp4 -p "a person"
```
- Draws a green box around the person's body/face
- Label: "a person" (detection mode) or detailed caption (caption mode)

#### Face Detection
```bash
python main.py -p "a person's face"
python main.py -f videos/Gus.mp4 -p "a person's face"
```
- Draws a box around the face region
- Label: "a person's face" (detection mode)

#### Object Detection
```bash
python main.py -p "a coffee mug"
python main.py -f videos/Gus.mp4 -p "a pair of glasses"
python main.py -f videos/Gus.mp4 -p "a laptop"
```
- Draws boxes around the specified objects
- Labels match the prompt text (in detection mode)

#### Caption Mode Examples
```bash
# Get detailed descriptions of detected people
python main.py -f videos/Gus.mp4 -p "a person" -m caption

# Get detailed descriptions of objects
python main.py -f videos/Gus.mp4 -p "a coffee mug" -m caption
python main.py -f videos/Gus.mp4 -p "a laptop" -m caption
```
- Draws boxes around detected objects
- Labels are AI-generated captions with rich details
- Example captions: "a person wearing a blue shirt standing in a room", "a white coffee mug on a wooden table"

#### General Scene Query
```bash
python main.py  # Uses default prompt
python main.py -f videos/Gus.mp4  # With video file from videos/ directory
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
