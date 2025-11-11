"""Pytest configuration and shared fixtures."""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, Mock
import numpy as np
from PIL import Image

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

import pytest


@pytest.fixture
def sample_frame():
    """Creates a sample video frame (numpy array) for testing.

    Returns:
        np.ndarray: A 720x1280x3 RGB frame.
    """
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def sample_pil_image():
    """Creates a sample PIL Image for testing.

    Returns:
        Image.Image: A 640x480 RGB image.
    """
    return Image.new("RGB", (640, 480), color="red")


@pytest.fixture
def sample_detected_objects():
    """Creates sample detected objects with bounding boxes.

    Returns:
        List[Dict]: List of detected objects with normalized coordinates.
    """
    return [
        {
            "x_min": 0.2,
            "y_min": 0.3,
            "x_max": 0.6,
            "y_max": 0.7,
            "label": "a person",
        }
    ]


@pytest.fixture
def mock_model():
    """Creates a mock Moondream model for testing.

    Returns:
        MagicMock: Mock model with detect and caption methods.
    """
    model = MagicMock()
    model.detect.return_value = {
        "objects": [
            {
                "x_min": 0.2,
                "y_min": 0.3,
                "x_max": 0.6,
                "y_max": 0.7,
            }
        ]
    }
    model.caption.return_value = {"caption": "a person wearing a blue shirt"}
    model.compile = MagicMock()
    return model


@pytest.fixture
def mock_video_capture():
    """Creates a mock OpenCV VideoCapture object.

    Returns:
        MagicMock: Mock VideoCapture with frame reading capabilities.
    """
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.return_value = 30.0  # FPS
    cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
    return cap


@pytest.fixture
def mock_video_writer():
    """Creates a mock OpenCV VideoWriter object.

    Returns:
        MagicMock: Mock VideoWriter for saving video.
    """
    writer = MagicMock()
    writer.write = MagicMock()
    writer.release = MagicMock()
    return writer

