import numpy as np

from src.policy.policy import Policy


class PolicyAloha(Policy):
    """Send packet with probability p"""

    def __init__(self, p_transmit: float = 0.1):
        super().__init__()
        self.p_transmit = p_transmit

    def action(self, observation):
        action = 0
        if observation.n_packets_buffer > 0:
            action = np.random.choice([0, 1], p=[1 - self.p_transmit, self.p_transmit])
        return action, None
