"""A thread-safe class for managing shared state between threads."""

import threading


class SharedState:
    """Manages a shared prompt with thread-safe access."""

    def __init__(self, initial_prompt: str):
        """Initializes the SharedState.

        Args:
            initial_prompt: The initial prompt to start with.
        """
        self._prompt = initial_prompt
        self._lock = threading.Lock()

    @property
    def prompt(self) -> str:
        """Gets the current prompt in a thread-safe way.

        Returns:
            The current prompt.
        """
        with self._lock:
            return self._prompt

    @prompt.setter
    def prompt(self, new_prompt: str):
        """Sets a new prompt in a thread-safe way.

        Args:
            new_prompt: The new prompt to set.
        """
        with self._lock:
            self._prompt = new_prompt
