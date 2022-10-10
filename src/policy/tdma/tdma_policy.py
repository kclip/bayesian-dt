import numpy as np

from src.policy.policy import Policy


class PolicyTDMA(Policy):
    """
    Send packet with probability <transmission_probability> at time steps
    <time_step> % <frame_length> = <transmission_slot>
    """

    def __init__(self, frame_length: int, transmission_slot: int, transmission_probability: float):
        super().__init__()
        self.frame_length = frame_length
        self.transmission_slot = transmission_slot
        self.transmission_probability = transmission_probability

    def action(self, observation):
        action = 0
        if (
            (observation.n_packets_buffer > 0) and
            (observation.time_step % self.frame_length == self.transmission_slot)
        ):
            action = np.random.choice([0, 1], p=[1 - self.transmission_probability, self.transmission_probability])
        return action, None
