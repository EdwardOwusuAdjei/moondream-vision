"""Handles the object detection model loading and inference.

This module provides a wrapper class for the Moondream model, simplifying the
process of loading the model and running object detection on images.
"""

import torch
from transformers import AutoModelForCausalLM
from PIL import Image
from typing import List, Dict, Any

from moondream_realtime_detector.config import settings


class Detector:
    """A wrapper for the Moondream object detection model."""

    def __init__(self):
        """Initializes the Detector and loads the Moondream model."""
        self.model = self._load_model()
        print("Compiling Moondream model...")
        self.model.compile()
        print("Moondream model loaded and compiled successfully.")

    def _load_model(self) -> AutoModelForCausalLM:
        """Loads the Moondream model from Hugging Face.

        Returns:
            The loaded Moondream model.
        """
        model = AutoModelForCausalLM.from_pretrained(
            settings.MODEL_ID,
            revision=settings.MODEL_REVISION,
            trust_remote_code=True,
            torch_dtype=settings.DTYPE,
        )
        model.to(settings.DEVICE)
        return model

    def detect_objects(
        self, image: Image.Image, prompt: str
    ) -> List[Dict[str, Any]]:
        """Detects objects in an image based on a text prompt.

        Args:
            image: The image in which to detect objects.
            prompt: The text prompt describing the objects to detect.

        Returns:
            A list of dictionaries, where each dictionary represents a detected
            object and contains its bounding box coordinates.
        """
        try:
            detection_settings = {"max_objects": settings.MAX_OBJECTS}
            result = self.model.detect(
                image, prompt, settings=detection_settings
            )
            return result.get("objects", [])
        except Exception as e:
            print(f"An error occurred during object detection: {e}")
            return []

    def caption_image(self, image: Image.Image) -> str:
        """Generates a caption for a given image.

        Args:
            image: The image to generate a caption for.

        Returns:
            The generated caption as a string.
        """
        try:
            result = self.model.caption(image, length="short")
            return result.get("caption", "")
        except Exception as e:
            print(f"An error occurred during image captioning: {e}")
            return ""
