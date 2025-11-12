"""Utility functions for drawing on images.

This module provides functions for drawing bounding boxes, labels, and other
informational text on image frames. Guys if you notice anything you can help improve this submit a PR.
"""

import cv2
import numpy as np
from typing import List, Dict, Any

from moondream_realtime_detector.config import settings


def _overlap(rect1: Dict, rect2: Dict) -> bool:
    """Check if two rectangles overlap."""
    return not (rect1['x_max'] < rect2['x_min'] or 
                rect2['x_max'] < rect1['x_min'] or 
                rect1['y_max'] < rect2['y_min'] or 
                rect2['y_max'] < rect1['y_min'])


def draw_bounding_boxes(
    frame: np.ndarray, objects: List[Dict[str, Any]]
) -> np.ndarray:
    """Draws bounding boxes and labels for detected objects on a frame.

    Args:
        frame: The image frame on which to draw.
        objects: A list of detected objects, each with bounding box coordinates
                 and a 'label' key.

    Returns:
        The frame with bounding boxes and labels drawn on it.
    """
    height, width, _ = frame.shape
    used_areas = []  # Track used text areas to prevent overlap
    
    for obj in objects:
        x_min = int(obj["x_min"] * width)
        y_min = int(obj["y_min"] * height)
        x_max = int(obj["x_max"] * width)
        y_max = int(obj["y_max"] * height)
        label = obj.get("label", "")

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x_min, y_min),
            (x_max, y_max),
            settings.BBOX_COLOR,
            settings.BBOX_THICKNESS,
        )

        # Draw label with "smart" wrapping to avoid overlap
        _draw_wrapped_text(frame, label, x_min, y_min, x_max, width, used_areas, height)

    return frame


def _draw_wrapped_text(
    frame: np.ndarray, text: str, x_min: int, y_min: int, x_max: int, frame_width: int, used_areas: List[Dict] = None, frame_height: int = None
):
    """Draws text with wrapping and a background within a bounding box."""
    font = settings.FONT
    font_scale = settings.FONT_SCALE
    thickness = settings.FONT_THICKNESS
    line_spacing = settings.TEXT_LINE_SPACING
    
    frame_height, frame_width_actual = frame.shape[:2]

    wrap_width = max(300, x_max - x_min)  # Minimum width for readability - no truncation
    words = text.split(" ")
    lines = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        (text_width, text_height), _ = cv2.getTextSize(
            test_line, font, font_scale, thickness
        )
        if text_width > wrap_width and current_line:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    
    if current_line:
        lines.append(current_line)
    
    # Handle empty text case
    if not lines:
        return
    
    # Don't truncate text - let it wrap naturally
    # We'll handle positioning to avoid overlap instead

    (line_width, line_height), _ = cv2.getTextSize(
        lines[0], font, font_scale, thickness
    )
    total_text_height = len(lines) * (line_height + line_spacing)

    label_x = x_min
    # Adjust label position if it goes off the right edge
    if (label_x + line_width) > frame_width:
        label_x = frame_width - line_width

    # Try to position text above the box first, then below if no space
    label_y_start = y_min - 10 if (y_min - 10 - total_text_height) > 0 else y_min + 10 + line_height
    
    # Check for overlap with other text areas if used_areas is provided
    if used_areas is not None and frame_height is not None:
        text_area = {
            'x_min': label_x - 15,
            'y_min': label_y_start - 5,
            'x_max': label_x + line_width + 15,
            'y_max': label_y_start + total_text_height + 5
        }
        
        # Check for overlap with existing text areas
        overlap_found = False
        for area in used_areas:
            if _overlap(text_area, area):
                overlap_found = True
                break
        
        # If overlap found, try positioning below the box
        if overlap_found and label_y_start > y_min:
            label_y_start = y_min + 10 + line_height
            text_area['y_min'] = label_y_start - 5
            text_area['y_max'] = label_y_start + total_text_height + 5
            
            # Check again for overlap below
            overlap_found = False
            for area in used_areas:
                if _overlap(text_area, area):
                    overlap_found = True
                    break
        
        # If still overlapping, try moving to the right
        if overlap_found:
            label_x = min(frame_width - line_width - 15, x_max + 10)
            text_area['x_min'] = label_x - 15
            text_area['x_max'] = label_x + line_width + 15
        
        # Add this text area to used areas
        used_areas.append(text_area)

    # Draw background for the text with proper bounds checking
    if settings.LABEL_BG_COLOR is not None and lines:
        # Calculate the actual width needed for the longest line
        max_width = 0
        for line in lines:
            (line_w, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_width = max(max_width, line_w)
        
        # Add extra padding to ensure text doesn't get cut off
        padding = 15
        
        if label_y_start > y_min:  # Text below box
            bg_y_start = max(0, y_min + 5)
            bg_y_end = min(frame_height, y_min + total_text_height + padding)
        else:  # Text above box
            bg_y_start = max(0, y_min - total_text_height - padding)
            bg_y_end = min(frame_height, y_min - 5)
        
        bg_x_start = max(0, label_x - padding)
        bg_x_end = min(frame_width, label_x + max_width + padding)
        
        cv2.rectangle(
            frame,
            (bg_x_start, bg_y_start),
            (bg_x_end, bg_y_end),
            settings.LABEL_BG_COLOR,
            cv2.FILLED,
        )

    # Draw each line of text with proper positioning
    for i, line in enumerate(lines):
        if label_y_start > y_min:  # If text is below the box
            y_pos = y_min + (i * (line_height + line_spacing)) + line_height + 10
        else:  # If text is above the box
            y_pos = y_min - total_text_height + (i * (line_height + line_spacing)) + line_height

        # Ensure text doesn't go off screen
        y_pos = max(line_height + 5, min(y_pos, frame_height - 5))

        # Draw text with outline for better visibility
        cv2.putText(
            frame,
            line,
            (label_x, y_pos),
            font,
            font_scale,
            (0, 0, 0),  # Black outline
            thickness + 1,
        )
        cv2.putText(
            frame,
            line,
            (label_x, y_pos),
            font,
            font_scale,
            settings.LABEL_COLOR,  # Green text
            thickness,
        )


def _draw_text_with_bg(
    frame: np.ndarray,
    text: str,
    position: tuple,
    font_color: tuple,
    bg_color: tuple,
):
    """Draws text with a padded background for maximum readability."""
    font = settings.FONT
    font_scale = settings.FONT_SCALE
    thickness = settings.FONT_THICKNESS
    padding = settings.TEXT_PADDING

    (text_width, text_height), _ = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    x, y = position
    frame_height, frame_width = frame.shape[:2]
    
    # Ensure text doesn't go off screen
    x = max(padding, min(x, frame_width - text_width - padding))
    y = max(text_height + padding, min(y, frame_height - padding))
    
    # Draw background rectangle for better readability
    if bg_color is not None:
        # Calculate background rectangle coordinates
        bg_x1 = max(0, x - padding)
        bg_y1 = max(0, y - text_height - padding)
        bg_x2 = min(frame_width, x + text_width + padding)
        bg_y2 = min(frame_height, y + padding)
        
        cv2.rectangle(
            frame,
            (bg_x1, bg_y1),
            (bg_x2, bg_y2),
            bg_color,
            cv2.FILLED,
        )
    
    # Draw text with outline for better visibility
    cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), thickness + 1)  # Black outline
    cv2.putText(frame, text, (x, y), font, font_scale, font_color, thickness)  # Green text


def draw_info_text(frame: np.ndarray, text: str) -> np.ndarray:
    """Draws informational text (like FPS) on a frame."""
    _draw_text_with_bg(
        frame,
        text,
        settings.INFO_TEXT_POSITION,
        settings.INFO_TEXT_COLOR,
        settings.INFO_BG_COLOR,
    )
    return frame


def draw_prompt_text(frame: np.ndarray, text: str) -> np.ndarray:
    """Draws the current prompt text on a frame."""
    _draw_text_with_bg(
        frame,
        f"Prompt: {text}",
        settings.PROMPT_TEXT_POSITION,
        settings.PROMPT_TEXT_COLOR,
        settings.PROMPT_BG_COLOR,
    )
    return frame
