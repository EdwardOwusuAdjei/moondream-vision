"""Configuration settings for the Moondream real-time object detector.

This module contains constants and configuration variables used throughout the
application.
"""

import torch

# Model Configuration
MODEL_ID = "vikhyatk/moondream2" #The idea is one day we can integrate other models but for now this is the only one we support.
MODEL_REVISION = "2025-01-09"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.bfloat16

# Detection Settings
DEFAULT_OBJECT_PROMPT = "what are the objects in the scene"
MAX_OBJECTS = 1  # Max number of objects to detect in a single frame
DEFAULT_MODE = "detect" # "detect" or "caption"

# Video Configuration
CAMERA_INDEX = 0  # 0 for default webcam
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS_POLL_RATE = 10 # Number of frames to average for FPS calculation
FRAME_SKIP = 5  # Process every 5th frame

# Display Configuration (so gonna add a little bit of color to the text)
WINDOW_NAME = "Surveillance Agent"
FONT = 0  # Using a simple OpenCV font
FONT_SCALE = 0.7
FONT_THICKNESS = 2
BBOX_COLOR = (0, 255, 0)  # Green
BBOX_THICKNESS = 2
LABEL_COLOR = (0, 255, 0)  # Green
INFO_TEXT_COLOR = (0, 255, 0) # Green
INFO_TEXT_POSITION = (10, FRAME_HEIGHT - 40)
PROMPT_TEXT_COLOR = (0, 255, 0) # Green
PROMPT_TEXT_POSITION = (10, FRAME_HEIGHT - 10)
LABEL_BG_COLOR = (0, 0, 0)  # Black background for readability
TEXT_LINE_SPACING = 5
INFO_BG_COLOR = (0, 0, 0)  # Black background for readability
PROMPT_BG_COLOR = (0, 0, 0)  # Black background for readability
TEXT_PADDING = 8  # Increased padding for better readability
