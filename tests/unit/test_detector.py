"""Unit tests for the Detector class."""

import pytest
from unittest.mock import patch, MagicMock, Mock
from PIL import Image

from moondream_realtime_detector.core.detector import Detector


class TestDetector:
    """Test suite for Detector class."""

    @patch("moondream_realtime_detector.core.detector.AutoModelForCausalLM")
    def test_detector_initialization(self, mock_model_class):
        """Test that Detector initializes and loads model correctly."""
        mock_model = MagicMock()
        mock_model.compile = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        detector = Detector()

        assert detector.model == mock_model
        mock_model_class.from_pretrained.assert_called_once()
        mock_model.compile.assert_called_once()

    @patch("moondream_realtime_detector.core.detector.AutoModelForCausalLM")
    def test_load_model_calls(self, mock_model_class):
        """Test that _load_model makes correct API calls."""
        mock_model = MagicMock()
        mock_model.compile = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        detector = Detector()

        # Verify from_pretrained was called with correct parameters
        call_kwargs = mock_model_class.from_pretrained.call_args[1]
        assert "trust_remote_code" in call_kwargs
        assert call_kwargs["trust_remote_code"] is True
        assert "torch_dtype" in call_kwargs

    @patch("moondream_realtime_detector.core.detector.AutoModelForCausalLM")
    def test_detect_objects_success(self, mock_model_class, sample_pil_image):
        """Test successful object detection."""
        mock_model = MagicMock()
        mock_model.compile = MagicMock()
        mock_model.detect.return_value = {
            "objects": [
                {
                    "x_min": 0.2,
                    "y_min": 0.3,
                    "x_max": 0.6,
                    "y_max": 0.7,
                }
            ]
        }
        mock_model_class.from_pretrained.return_value = mock_model

        detector = Detector()
        result = detector.detect_objects(sample_pil_image, "a person")

        assert len(result) == 1
        assert "x_min" in result[0]
        assert "y_min" in result[0]
        assert "x_max" in result[0]
        assert "y_max" in result[0]
        mock_model.detect.assert_called_once()

    @patch("moondream_realtime_detector.core.detector.AutoModelForCausalLM")
    def test_detect_objects_empty_result(self, mock_model_class, sample_pil_image):
        """Test object detection with empty result."""
        mock_model = MagicMock()
        mock_model.compile = MagicMock()
        mock_model.detect.return_value = {"objects": []}
        mock_model_class.from_pretrained.return_value = mock_model

        detector = Detector()
        result = detector.detect_objects(sample_pil_image, "a person")

        assert result == []

    @patch("moondream_realtime_detector.core.detector.AutoModelForCausalLM")
    def test_detect_objects_exception_handling(self, mock_model_class, sample_pil_image):
        """Test that exceptions during detection are handled gracefully."""
        mock_model = MagicMock()
        mock_model.compile = MagicMock()
        mock_model.detect.side_effect = Exception("Detection error")
        mock_model_class.from_pretrained.return_value = mock_model

        detector = Detector()
        result = detector.detect_objects(sample_pil_image, "a person")

        assert result == []

    @patch("moondream_realtime_detector.core.detector.AutoModelForCausalLM")
    def test_detect_objects_with_max_objects_setting(self, mock_model_class, sample_pil_image):
        """Test that max_objects setting is passed to model."""
        mock_model = MagicMock()
        mock_model.compile = MagicMock()
        mock_model.detect.return_value = {"objects": []}
        mock_model_class.from_pretrained.return_value = mock_model

        detector = Detector()
        detector.detect_objects(sample_pil_image, "a person")

        # Verify detect was called with settings
        call_kwargs = mock_model.detect.call_args[1]
        assert "settings" in call_kwargs
        assert "max_objects" in call_kwargs["settings"]

    @patch("moondream_realtime_detector.core.detector.AutoModelForCausalLM")
    def test_caption_image_success(self, mock_model_class, sample_pil_image):
        """Test successful image captioning."""
        mock_model = MagicMock()
        mock_model.compile = MagicMock()
        mock_model.caption.return_value = {"caption": "a person wearing a blue shirt"}
        mock_model_class.from_pretrained.return_value = mock_model

        detector = Detector()
        result = detector.caption_image(sample_pil_image)

        assert result == "a person wearing a blue shirt"
        mock_model.caption.assert_called_once()

    @patch("moondream_realtime_detector.core.detector.AutoModelForCausalLM")
    def test_caption_image_empty_result(self, mock_model_class, sample_pil_image):
        """Test captioning with empty result."""
        mock_model = MagicMock()
        mock_model.compile = MagicMock()
        mock_model.caption.return_value = {"caption": ""}
        mock_model_class.from_pretrained.return_value = mock_model

        detector = Detector()
        result = detector.caption_image(sample_pil_image)

        assert result == ""

    @patch("moondream_realtime_detector.core.detector.AutoModelForCausalLM")
    def test_caption_image_exception_handling(self, mock_model_class, sample_pil_image):
        """Test that exceptions during captioning are handled gracefully."""
        mock_model = MagicMock()
        mock_model.compile = MagicMock()
        mock_model.caption.side_effect = Exception("Caption error")
        mock_model_class.from_pretrained.return_value = mock_model

        detector = Detector()
        result = detector.caption_image(sample_pil_image)

        assert result == ""

    @patch("moondream_realtime_detector.core.detector.AutoModelForCausalLM")
    def test_caption_image_with_length_parameter(self, mock_model_class, sample_pil_image):
        """Test that caption is called with length parameter."""
        mock_model = MagicMock()
        mock_model.compile = MagicMock()
        mock_model.caption.return_value = {"caption": "test"}
        mock_model_class.from_pretrained.return_value = mock_model

        detector = Detector()
        detector.caption_image(sample_pil_image)

        # Verify caption was called with length="short"
        call_kwargs = mock_model.caption.call_args[1]
        assert call_kwargs["length"] == "short"

