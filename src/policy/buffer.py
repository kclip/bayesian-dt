from typing import List
from collections import deque

from src.data_classes import Transition


class CircularBuffer(object):
    def __init__(self, max_size: int):
        self.max_size = max_size
        self._buffer = deque([], maxlen=self.max_size)  # Once max_size is reached, older entries are deleted

    @property
    def buffer(self):
        return self._buffer

    @property
    def buffer_values(self):
        return list(self._buffer)

    def append(self, element):
        self._buffer.append(element)

    def reset(self):
        self._buffer = deque([], maxlen=self.max_size)

    def __len__(self):
        return len(self._buffer)


class NStepsBuffer(CircularBuffer):
    def append(self, transitions: List[Transition]):
        for t in transitions:
            super().append(t)

    def remove_older_entries(self, n_entries: int):
        for _ in range(min(n_entries, len(self._buffer))):
            del self._buffer[0]
