import random
from collections import deque
from typing import List

from src.data_classes import AgentTransition


class ReplayBuffer(object):
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.memory = deque([], maxlen=max_size)  # Once max_size is reached, older entries are deleted

    def append(self, transition: AgentTransition):
        self.memory.append(transition)

    def sample(self, batch_size: int) -> List[AgentTransition]:
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
