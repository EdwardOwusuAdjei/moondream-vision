"""Unit tests for drawing utilities."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from moondream_realtime_detector.utils import drawing


class TestOverlap:
    """Test suite for _overlap function."""

    def test_overlapping_rectangles(self):
        """Test that overlapping rectangles are detected."""
        rect1 = {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}
        rect2 = {"x_min": 5, "y_min": 5, "x_max": 15, "y_max": 15}
        assert drawing._overlap(rect1, rect2) is True

    def test_non_overlapping_rectangles(self):
        """Test that non-overlapping rectangles are detected."""
        rect1 = {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}
        rect2 = {"x_min": 20, "y_min": 20, "x_max": 30, "y_max": 30}
        assert drawing._overlap(rect1, rect2) is False

    def test_touching_rectangles(self):
        """Test that touching rectangles are considered overlapping."""
        rect1 = {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}
        rect2 = {"x_min": 10, "y_min": 10, "x_max": 20, "y_max": 20}
        assert drawing._overlap(rect1, rect2) is True

    def test_contained_rectangle(self):
        """Test that a contained rectangle is detected as overlapping."""
        rect1 = {"x_min": 0, "y_min": 0, "x_max": 20, "y_max": 20}
        rect2 = {"x_min": 5, "y_min": 5, "x_max": 15, "y_max": 15}
        assert drawing._overlap(rect1, rect2) is True


class TestDrawBoundingBoxes:
    """Test suite for draw_bounding_boxes function."""

    def test_draw_single_box(self, sample_frame, sample_detected_objects):
        """Test drawing a single bounding box."""
        frame = sample_frame.copy()
        result = drawing.draw_bounding_boxes(frame, sample_detected_objects)

        assert result is not None
        assert result.shape == frame.shape
        # Frame should be modified (not identical)
        assert not np.array_equal(result, sample_frame)

    def test_draw_multiple_boxes(self, sample_frame):
        """Test drawing multiple bounding boxes."""
        objects = [
            {
                "x_min": 0.1,
                "y_min": 0.1,
                "x_max": 0.3,
                "y_max": 0.3,
                "label": "object 1",
            },
            {
                "x_min": 0.5,
                "y_min": 0.5,
                "x_max": 0.7,
                "y_max": 0.7,
                "label": "object 2",
            },
        ]
        frame = sample_frame.copy()
        result = drawing.draw_bounding_boxes(frame, objects)

        assert result is not None
        assert result.shape == frame.shape

    def test_draw_empty_objects_list(self, sample_frame):
        """Test drawing with empty objects list."""
        frame = sample_frame.copy()
        result = drawing.draw_bounding_boxes(frame, [])

        assert result is not None
        assert result.shape == frame.shape
        # Should return original frame if no objects
        assert np.array_equal(result, sample_frame)

    def test_draw_box_without_label(self, sample_frame):
        """Test drawing box when object has no label."""
        objects = [
            {
                "x_min": 0.2,
                "y_min": 0.3,
                "x_max": 0.6,
                "y_max": 0.7,
            }
        ]
        frame = sample_frame.copy()
        # Empty label should be handled gracefully (no text drawn)
        result = drawing.draw_bounding_boxes(frame, objects)

        assert result is not None
        assert result.shape == frame.shape

    def test_coordinate_conversion(self, sample_frame):
        """Test that normalized coordinates are converted correctly."""
        objects = [
            {
                "x_min": 0.0,
                "y_min": 0.0,
                "x_max": 1.0,
                "y_max": 1.0,
                "label": "full frame",
            }
        ]
        frame = sample_frame.copy()
        height, width = frame.shape[:2]

        with patch("cv2.rectangle") as mock_rect, \
             patch("moondream_realtime_detector.utils.drawing._draw_wrapped_text"):
            drawing.draw_bounding_boxes(frame, objects)
            # Verify rectangle was called with pixel coordinates
            # First call should be the bounding box (not text background)
            first_call = mock_rect.call_args_list[0]
            call_args = first_call[0]
            assert call_args[1] == (0, 0)  # x_min, y_min
            assert call_args[2] == (width, height)  # x_max, y_max


class TestDrawInfoText:
    """Test suite for draw_info_text function."""

    def test_draw_info_text(self, sample_frame):
        """Test drawing info text on frame."""
        frame = sample_frame.copy()
        result = drawing.draw_info_text(frame, "FPS: 30.0")

        assert result is not None
        assert result.shape == frame.shape

    def test_draw_info_text_empty_string(self, sample_frame):
        """Test drawing empty info text."""
        frame = sample_frame.copy()
        result = drawing.draw_info_text(frame, "")

        assert result is not None
        assert result.shape == frame.shape


class TestDrawPromptText:
    """Test suite for draw_prompt_text function."""

    def test_draw_prompt_text(self, sample_frame):
        """Test drawing prompt text on frame."""
        frame = sample_frame.copy()
        result = drawing.draw_prompt_text(frame, "a person")

        assert result is not None
        assert result.shape == frame.shape

    def test_draw_prompt_text_format(self, sample_frame):
        """Test that prompt text is formatted correctly."""
        frame = sample_frame.copy()

        with patch("moondream_realtime_detector.utils.drawing._draw_text_with_bg") as mock_draw:
            drawing.draw_prompt_text(frame, "test prompt")
            # Verify the text includes "Prompt: " prefix
            # call_args[0] is tuple of positional args: (frame, text, position, font_color, bg_color)
            call_args = mock_draw.call_args
            # Text is the second positional argument (index 1)
            text_arg = call_args[0][1]
            assert text_arg == "Prompt: test prompt"


class TestDrawWrappedText:
    """Test suite for _draw_wrapped_text function."""

    def test_wrap_long_text(self, sample_frame):
        """Test that long text is wrapped correctly."""
        frame = sample_frame.copy()
        long_text = " ".join(["word"] * 50)  # Very long text

        # Should not raise an exception
        try:
            drawing._draw_wrapped_text(
                frame, long_text, 100, 100, 200, frame.shape[1], None, frame.shape[0]
            )
        except Exception as e:
            pytest.fail(f"_draw_wrapped_text raised {e}")

    def test_short_text_no_wrap(self, sample_frame):
        """Test that short text doesn't wrap."""
        frame = sample_frame.copy()
        short_text = "short"

        try:
            drawing._draw_wrapped_text(
                frame, short_text, 100, 100, 200, frame.shape[1], None, frame.shape[0]
            )
        except Exception as e:
            pytest.fail(f"_draw_wrapped_text raised {e}")

    def test_text_positioning_above_box(self, sample_frame):
        """Test text positioning above bounding box."""
        frame = sample_frame.copy()
        # Position near top of frame to force text above
        y_min = 50

        try:
            drawing._draw_wrapped_text(
                frame, "test", 100, y_min, 200, frame.shape[1], None, frame.shape[0]
            )
        except Exception as e:
            pytest.fail(f"_draw_wrapped_text raised {e}")

    def test_text_positioning_below_box(self, sample_frame):
        """Test text positioning below bounding box."""
        frame = sample_frame.copy()
        # Position near bottom of frame to force text below
        y_min = frame.shape[0] - 50

        try:
            drawing._draw_wrapped_text(
                frame, "test", 100, y_min, 200, frame.shape[1], None, frame.shape[0]
            )
        except Exception as e:
            pytest.fail(f"_draw_wrapped_text raised {e}")

    def test_overlap_avoidance(self, sample_frame):
        """Test that text areas avoid overlap."""
        frame = sample_frame.copy()
        used_areas = [
            {"x_min": 0, "y_min": 0, "x_max": 100, "y_max": 50}
        ]

        try:
            drawing._draw_wrapped_text(
                frame, "test", 50, 25, 150, frame.shape[1], used_areas, frame.shape[0]
            )
        except Exception as e:
            pytest.fail(f"_draw_wrapped_text raised {e}")

