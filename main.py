"""Main entry point for the Moondream real-time object detector application.

This script initializes and runs the video processor, which captures video
from the webcam, detects objects based on a predefined prompt, and displays
the results in real-time.
"""

import argparse
import sys
import os
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from moondream_realtime_detector.config import settings
from moondream_realtime_detector.core.video_processor import VideoProcessor
from moondream_realtime_detector.utils.state import SharedState


def prompt_input_thread(shared_state: SharedState):
    """Listens for user input to update the prompt."""
    print("\nPress Enter to change the prompt. Type 'quit' to exit.")
    while True:
        try:
            new_prompt = input("New prompt: ")
            if new_prompt.lower() == "quit":
                break
            if new_prompt:
                shared_state.prompt = new_prompt
                print(f"Prompt updated to: '{new_prompt}'")
        except EOFError:
            # This can happen if the main thread exits
            break


def main(args):
    """Initializes and runs the VideoProcessor.

    Args:
        args: Command-line arguments.
    """
    print("Starting Moondream real-time object detector...")
    initial_prompt = args.prompt or settings.DEFAULT_OBJECT_PROMPT
    mode = args.mode or settings.DEFAULT_MODE
    
    shared_state = SharedState(initial_prompt)

    # Start the input listener thread
    input_thread = threading.Thread(
        target=prompt_input_thread, args=(shared_state,), daemon=True
    )
    input_thread.start()
    
    print(f"Initial prompt: '{initial_prompt}'")
    print(f"Running in mode: '{mode}'")

    try:
        video_processor = VideoProcessor(
            shared_state=shared_state,
            mode=mode,
            filepath=args.filepath,
            output_path=args.output,
        )
        video_processor.run()
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Moondream Real-time Object Detector"
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="The object detection prompt.",
        default=settings.DEFAULT_OBJECT_PROMPT,
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["detect", "caption"],
        help="The operation mode: 'detect' or 'caption'.",
        default=settings.DEFAULT_MODE,
    )
    parser.add_argument(
        "-f",
        "--filepath",
        type=str,
        help="The path to an MP4 video file to process.",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="The path to save the output video file.",
        default=None,
    )
    args = parser.parse_args()
    main(args)
