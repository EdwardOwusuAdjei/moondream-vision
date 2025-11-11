"""Integration tests for VideoProcessor class."""

import pytest
from unittest.mock import patch, MagicMock, Mock
import numpy as np
import cv2

from moondream_realtime_detector.core.video_processor import VideoProcessor
from moondream_realtime_detector.utils.state import SharedState


class TestVideoProcessor:
    """Test suite for VideoProcessor class."""

    @patch("moondream_realtime_detector.core.video_processor.cv2.VideoCapture")
    @patch("moondream_realtime_detector.core.video_processor.Detector")
    def test_video_processor_initialization_webcam(
        self, mock_detector_class, mock_video_capture_class, mock_video_capture
    ):
        """Test VideoProcessor initialization with webcam."""
        mock_video_capture_class.return_value = mock_video_capture
        mock_video_capture.isOpened.return_value = True
        mock_video_capture.get.return_value = 30.0
        mock_detector = MagicMock()
        mock_detector_class.return_value = mock_detector

        shared_state = SharedState("test prompt")
        processor = VideoProcessor(shared_state=shared_state, mode="detect")

        assert processor.shared_state == shared_state
        assert processor.mode == "detect"
        assert processor.detector == mock_detector
        assert processor.is_file is False

    @patch("moondream_realtime_detector.core.video_processor.cv2.VideoCapture")
    @patch("moondream_realtime_detector.core.video_processor.Detector")
    def test_video_processor_initialization_file(
        self, mock_detector_class, mock_video_capture_class, mock_video_capture
    ):
        """Test VideoProcessor initialization with video file."""
        mock_video_capture_class.return_value = mock_video_capture
        mock_video_capture.isOpened.return_value = True
        mock_video_capture.get.return_value = 30.0
        mock_detector = MagicMock()
        mock_detector_class.return_value = mock_detector

        shared_state = SharedState("test prompt")
        processor = VideoProcessor(
            shared_state=shared_state,
            mode="detect",
            filepath="test_video.mp4",
        )

        assert processor.is_file is True
        assert processor.filepath == "test_video.mp4"

    @patch("moondream_realtime_detector.core.video_processor.cv2.VideoCapture")
    @patch("moondream_realtime_detector.core.video_processor.Detector")
    def test_video_processor_initialization_failure(
        self, mock_detector_class, mock_video_capture_class, mock_video_capture
    ):
        """Test VideoProcessor initialization failure when camera can't open."""
        mock_video_capture_class.return_value = mock_video_capture
        mock_video_capture.isOpened.return_value = False
        mock_detector = MagicMock()
        mock_detector_class.return_value = mock_detector

        shared_state = SharedState("test prompt")

        with pytest.raises(IOError):
            VideoProcessor(shared_state=shared_state, mode="detect")

    @patch("moondream_realtime_detector.core.video_processor.cv2.VideoCapture")
    @patch("moondream_realtime_detector.core.video_processor.Detector")
    @patch("moondream_realtime_detector.core.video_processor.cv2.VideoWriter")
    def test_video_writer_initialization(
        self,
        mock_video_writer_class,
        mock_detector_class,
        mock_video_capture_class,
        mock_video_capture,
    ):
        """Test VideoWriter initialization when output path is provided."""
        mock_video_capture_class.return_value = mock_video_capture
        mock_video_capture.isOpened.return_value = True
        mock_video_capture.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1280,
            cv2.CAP_PROP_FRAME_HEIGHT: 720,
            cv2.CAP_PROP_FPS: 30.0,
        }.get(prop, 0)
        mock_detector = MagicMock()
        mock_detector_class.return_value = mock_detector
        mock_video_writer = MagicMock()
        mock_video_writer_class.return_value = mock_video_writer

        shared_state = SharedState("test prompt")
        processor = VideoProcessor(
            shared_state=shared_state,
            mode="detect",
            output_path="output.mp4",
        )

        assert processor.video_writer is not None
        mock_video_writer_class.assert_called_once()

    @patch("moondream_realtime_detector.core.video_processor.cv2.VideoCapture")
    @patch("moondream_realtime_detector.core.video_processor.Detector")
    @patch("moondream_realtime_detector.core.video_processor.drawing")
    def test_process_frame_detection_mode(
        self,
        mock_drawing,
        mock_detector_class,
        mock_video_capture_class,
        mock_video_capture,
        sample_frame,
    ):
        """Test frame processing in detection mode."""
        mock_video_capture_class.return_value = mock_video_capture
        mock_video_capture.isOpened.return_value = True
        mock_video_capture.get.return_value = 30.0

        mock_detector = MagicMock()
        mock_detector.detect_objects.return_value = [
            {
                "x_min": 0.2,
                "y_min": 0.3,
                "x_max": 0.6,
                "y_max": 0.7,
                "label": "a person",
            }
        ]
        mock_detector_class.return_value = mock_detector

        mock_drawing.draw_bounding_boxes.return_value = sample_frame
        mock_drawing.draw_info_text.return_value = sample_frame
        mock_drawing.draw_prompt_text.return_value = sample_frame

        shared_state = SharedState("a person")
        processor = VideoProcessor(shared_state=shared_state, mode="detect")
        processor.frame_count = 0  # Force detection on first frame

        result = processor._process_frame(sample_frame.copy())

        assert result is not None
        mock_detector.detect_objects.assert_called_once()
        mock_drawing.draw_bounding_boxes.assert_called_once()
        mock_drawing.draw_info_text.assert_called_once()
        mock_drawing.draw_prompt_text.assert_called_once()

    @patch("moondream_realtime_detector.core.video_processor.cv2.VideoCapture")
    @patch("moondream_realtime_detector.core.video_processor.Detector")
    @patch("moondream_realtime_detector.core.video_processor.drawing")
    def test_process_frame_caption_mode(
        self,
        mock_drawing,
        mock_detector_class,
        mock_video_capture_class,
        mock_video_capture,
        sample_frame,
    ):
        """Test frame processing in caption mode."""
        mock_video_capture_class.return_value = mock_video_capture
        mock_video_capture.isOpened.return_value = True
        mock_video_capture.get.return_value = 30.0

        mock_detector = MagicMock()
        mock_detector.detect_objects.return_value = [
            {
                "x_min": 0.2,
                "y_min": 0.3,
                "x_max": 0.6,
                "y_max": 0.7,
            }
        ]
        mock_detector.caption_image.return_value = "a person wearing a blue shirt"
        mock_detector_class.return_value = mock_detector

        mock_drawing.draw_bounding_boxes.return_value = sample_frame
        mock_drawing.draw_info_text.return_value = sample_frame
        mock_drawing.draw_prompt_text.return_value = sample_frame

        shared_state = SharedState("a person")
        processor = VideoProcessor(shared_state=shared_state, mode="caption")
        processor.frame_count = 0  # Force detection on first frame

        result = processor._process_frame(sample_frame.copy())

        assert result is not None
        mock_detector.detect_objects.assert_called_once()
        mock_detector.caption_image.assert_called_once()
        # Verify caption was used as label
        call_args = mock_drawing.draw_bounding_boxes.call_args
        objects = call_args[0][1]
        assert objects[0]["label"] == "a person wearing a blue shirt"

    @patch("moondream_realtime_detector.core.video_processor.cv2.VideoCapture")
    @patch("moondream_realtime_detector.core.video_processor.Detector")
    @patch("moondream_realtime_detector.core.video_processor.drawing")
    def test_process_frame_skip_detection(
        self,
        mock_drawing,
        mock_detector_class,
        mock_video_capture_class,
        mock_video_capture,
        sample_frame,
    ):
        """Test that detection is skipped on non-processed frames."""
        mock_video_capture_class.return_value = mock_video_capture
        mock_video_capture.isOpened.return_value = True
        mock_video_capture.get.return_value = 30.0

        mock_detector = MagicMock()
        mock_detector_class.return_value = mock_detector

        mock_drawing.draw_bounding_boxes.return_value = sample_frame
        mock_drawing.draw_info_text.return_value = sample_frame
        mock_drawing.draw_prompt_text.return_value = sample_frame

        shared_state = SharedState("a person")
        processor = VideoProcessor(shared_state=shared_state, mode="detect")
        processor.frame_count = 1  # Skip detection (not divisible by FRAME_SKIP)
        processor.latest_objects = []  # No previous detections

        result = processor._process_frame(sample_frame.copy())

        assert result is not None
        # Detection should not be called
        mock_detector.detect_objects.assert_not_called()

    @patch("moondream_realtime_detector.core.video_processor.cv2.VideoCapture")
    @patch("moondream_realtime_detector.core.video_processor.Detector")
    @patch("moondream_realtime_detector.core.video_processor.cv2.imshow")
    @patch("moondream_realtime_detector.core.video_processor.cv2.waitKey")
    def test_run_loop_exit_on_q_key(
        self,
        mock_wait_key,
        mock_imshow,
        mock_detector_class,
        mock_video_capture_class,
        mock_video_capture,
        sample_frame,
    ):
        """Test that run loop exits when 'q' key is pressed."""
        mock_video_capture_class.return_value = mock_video_capture
        mock_video_capture.isOpened.return_value = True
        mock_video_capture.get.return_value = 30.0
        mock_video_capture.read.return_value = (True, sample_frame)

        mock_detector = MagicMock()
        mock_detector_class.return_value = mock_detector

        # Simulate 'q' key press
        mock_wait_key.return_value = ord("q")

        shared_state = SharedState("test")
        processor = VideoProcessor(shared_state=shared_state, mode="detect")

        # Mock the _process_frame to avoid actual processing
        processor._process_frame = MagicMock(return_value=sample_frame)

        processor.run()

        # Verify video capture was read
        assert mock_video_capture.read.called
        # Verify waitKey was called
        assert mock_wait_key.called

    @patch("moondream_realtime_detector.core.video_processor.cv2.VideoCapture")
    @patch("moondream_realtime_detector.core.video_processor.Detector")
    def test_cleanup_releases_resources(
        self, mock_detector_class, mock_video_capture_class, mock_video_capture
    ):
        """Test that cleanup releases all resources."""
        mock_video_capture_class.return_value = mock_video_capture
        mock_video_capture.isOpened.return_value = True
        mock_video_capture.get.return_value = 30.0
        mock_video_capture.release = MagicMock()

        mock_detector = MagicMock()
        mock_detector_class.return_value = mock_detector

        shared_state = SharedState("test")
        processor = VideoProcessor(shared_state=shared_state, mode="detect")
        processor.video_writer = MagicMock()
        processor.video_writer.release = MagicMock()

        processor._cleanup()

        mock_video_capture.release.assert_called_once()
        processor.video_writer.release.assert_called_once()

    @patch("moondream_realtime_detector.core.video_processor.cv2.VideoCapture")
    @patch("moondream_realtime_detector.core.video_processor.Detector")
    def test_calculate_fps(
        self, mock_detector_class, mock_video_capture_class, mock_video_capture
    ):
        """Test FPS calculation."""
        mock_video_capture_class.return_value = mock_video_capture
        mock_video_capture.isOpened.return_value = True
        mock_video_capture.get.return_value = 30.0

        mock_detector = MagicMock()
        mock_detector_class.return_value = mock_detector

        shared_state = SharedState("test")
        processor = VideoProcessor(shared_state=shared_state, mode="detect")

        # Set initial state
        processor.frame_count = 0
        processor.fps = 0

        # Calculate FPS after processing frames
        processor.frame_count = 10
        processor._calculate_fps()

        # FPS should be calculated (exact value depends on timing)
        assert processor.frame_count == 0  # Reset after calculation

