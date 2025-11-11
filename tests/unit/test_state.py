"""Unit tests for the SharedState class."""

import threading
import time
import pytest

from moondream_realtime_detector.utils.state import SharedState


class TestSharedState:
    """Test suite for SharedState class."""

    def test_initialization(self):
        """Test that SharedState initializes with correct prompt."""
        initial_prompt = "test prompt"
        state = SharedState(initial_prompt)
        assert state.prompt == initial_prompt

    def test_prompt_getter(self):
        """Test that prompt getter returns the current prompt."""
        state = SharedState("initial")
        assert state.prompt == "initial"

    def test_prompt_setter(self):
        """Test that prompt setter updates the prompt."""
        state = SharedState("initial")
        new_prompt = "new prompt"
        state.prompt = new_prompt
        assert state.prompt == new_prompt

    def test_thread_safety_concurrent_reads(self):
        """Test that multiple threads can read concurrently."""
        state = SharedState("initial")
        results = []

        def read_prompt():
            results.append(state.prompt)

        threads = [threading.Thread(target=read_prompt) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(results) == 10
        assert all(result == "initial" for result in results)

    def test_thread_safety_concurrent_writes(self):
        """Test that concurrent writes are handled safely."""
        state = SharedState("initial")
        num_threads = 10

        def write_prompt(thread_id: int):
            state.prompt = f"prompt_{thread_id}"
            time.sleep(0.01)  # Simulate some work
            # Verify the value was set (may be overwritten by other threads)
            assert state.prompt.startswith("prompt_")

        threads = [
            threading.Thread(target=write_prompt, args=(i,))
            for i in range(num_threads)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Final value should be one of the written values
        assert state.prompt.startswith("prompt_")

    def test_thread_safety_read_write_mix(self):
        """Test that reads and writes can happen concurrently."""
        state = SharedState("initial")
        read_results = []
        write_complete = threading.Event()

        def reader():
            while not write_complete.is_set():
                read_results.append(state.prompt)
                time.sleep(0.001)

        def writer():
            for i in range(5):
                state.prompt = f"updated_{i}"
                time.sleep(0.01)
            write_complete.set()

        reader_thread = threading.Thread(target=reader)
        writer_thread = threading.Thread(target=writer)

        reader_thread.start()
        writer_thread.start()

        writer_thread.join()
        time.sleep(0.05)  # Give reader time to finish
        write_complete.set()
        reader_thread.join(timeout=1.0)

        # Should have read some values
        assert len(read_results) > 0
        # All reads should be valid prompts
        assert all(
            result == "initial" or result.startswith("updated_")
            for result in read_results
        )

    def test_empty_prompt(self):
        """Test that empty prompt is handled correctly."""
        state = SharedState("")
        assert state.prompt == ""
        state.prompt = "new"
        assert state.prompt == "new"

    def test_long_prompt(self):
        """Test that long prompts are handled correctly."""
        long_prompt = "a" * 1000
        state = SharedState(long_prompt)
        assert state.prompt == long_prompt
        assert len(state.prompt) == 1000

