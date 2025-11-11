"""Handles video processing for real-time object detection.

This module contains the VideoProcessor class, which is responsible for
capturing video from a webcam, processing each frame to detect objects,
and displaying the results in real-time.
"""

import cv2
import time
from PIL import Image
from typing import TYPE_CHECKING

from moondream_realtime_detector.config import settings
from moondream_realtime_detector.core.detector import Detector
from moondream_realtime_detector.utils import drawing


class VideoProcessor:
    """Processes video stream for real-time object detection."""

    def __init__(
        self,
        shared_state: "SharedState",
        mode: str,
        filepath: str | None = None,
        output_path: str | None = None,
    ):
        """Initializes the VideoProcessor.

        Args:
            shared_state: The shared state object for the prompt.
            mode: The operation mode ('detect' or 'caption').
            filepath: The path to a video file to process. If None, uses the webcam.
            output_path: The path to save the output video file.
        """
        self.shared_state = shared_state
        self.mode = mode
        self.detector = Detector()
        self.filepath = filepath
        self.is_file = self.filepath is not None
        self.output_path = output_path
        self.video_writer = None

        source = self.filepath if self.is_file else settings.CAMERA_INDEX
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise IOError(
                f"Cannot open video source: {source}"
            )

        if not self.is_file:
            self._configure_camera()
        
        if self.output_path:
            self._initialize_video_writer()

        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_delay = int(1000 / self.video_fps) if self.is_file and self.video_fps > 0 else 1

        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        self.latest_objects = []

    def _initialize_video_writer(self):
        """Initializes the VideoWriter object for saving the video."""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:  # Webcam might return 0
            fps = 30 # Default to 30 FPS for webcam recording
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            self.output_path, fourcc, fps, (width, height)
        )
        print(f"Output video will be saved to: {self.output_path}")

    def _configure_camera(self):
        """Configures camera settings like resolution."""
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)

    def _process_frame(self, frame: "np.ndarray"):
        """Processes a single frame from the video stream."""
        # Only run detection every N frames
        if self.frame_count % settings.FRAME_SKIP == 0:
            # Get the latest prompt
            current_prompt = self.shared_state.prompt

            # Convert the frame to PIL Image for the model
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # Detect objects
            detected_objects = self.detector.detect_objects(pil_image, current_prompt)
            
            processed_objects = []
            for obj in detected_objects:
                label = current_prompt
                if self.mode == "caption":
                    # Crop the image to the bounding box
                    height, width, _ = frame.shape
                    x_min = int(obj["x_min"] * width)
                    y_min = int(obj["y_min"] * height)
                    x_max = int(obj["x_max"] * width)
                    y_max = int(obj["y_max"] * height)
                    
                    cropped_image = pil_image.crop((x_min, y_min, x_max, y_max))

                    # Caption the cropped image
                    if cropped_image.size[0] > 0 and cropped_image.size[1] > 0:
                        label = self.detector.caption_image(cropped_image)

                obj["label"] = label
                processed_objects.append(obj)
            
            self.latest_objects = processed_objects

        # Draw bounding boxes from the latest detection
        frame = drawing.draw_bounding_boxes(frame, self.latest_objects)
        
        # Calculate and draw FPS
        self._calculate_fps()
        info_text = f"FPS: {self.fps:.2f}"
        frame = drawing.draw_info_text(frame, info_text)

        # Draw the current prompt
        frame = drawing.draw_prompt_text(frame, self.shared_state.prompt)

        return frame

    def _calculate_fps(self):
        """Calculates and updates the FPS."""
        self.frame_count += 1
        if self.frame_count >= settings.FPS_POLL_RATE:
            end_time = time.time()
            elapsed_time = end_time - self.start_time
            if elapsed_time > 0:
                self.fps = self.frame_count / elapsed_time
            self.start_time = end_time
            self.frame_count = 0

    def run(self):
        """Starts the video processing loop."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("End of video file or failed to grab frame.")
                break

            processed_frame = self._process_frame(frame)
            if processed_frame is not None:
                cv2.imshow(settings.WINDOW_NAME, processed_frame)
                if self.video_writer:
                    self.video_writer.write(processed_frame)

            key = cv2.waitKey(self.frame_delay) & 0xFF
            if key == ord("q"):
                break

        self._cleanup()

    def _cleanup(self):
        """Releases resources and closes windows."""
        if self.video_writer:
            self.video_writer.release()
            print("Output video saved successfully.")
        self.cap.release()
        cv2.destroyAllWindows()
        print("Video processing finished.")
