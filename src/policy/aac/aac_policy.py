import logging
import torch
import numpy as np

from src.policy.policy import Policy
from src.policy.aac.utils import format_observation


logger = logging.getLogger(__name__)


class PolicyCommonAAC(Policy):
    def __init__(self, policy_nn, value_nn, n_packets_max: int):
        super().__init__()
        self.policy_nn = policy_nn
        self.value_nn = value_nn  # For monitoring purposes only
        self.n_packets_max = n_packets_max

    def get_value_critic(self, observation) -> float:
        with torch.no_grad():
            return self.value_nn(format_observation(observation, self.n_packets_max)).item()

    def get_p_transmit(self, observation) -> float:
        with torch.no_grad():
            return self.policy_nn(format_observation(observation, self.n_packets_max)).item()

    def action(self, observation):
        if observation.n_packets_buffer == 0:
            return 0, None
        with torch.no_grad():  # Gradients are only computed during replay
            p_transmit = self.get_p_transmit(observation)
            action = np.random.choice([0, 1], p=[1 - p_transmit, p_transmit])
        return action, None