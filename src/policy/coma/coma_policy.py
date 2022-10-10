import logging
import torch
import numpy as np

from src.data_classes import Observation
from src.policy.policy import Policy
from src.policy.coma.utils import format_actor_input


logger = logging.getLogger(__name__)


class PolicyCOMA(Policy):
    def __init__(self, policy_nn, n_packets_max: int, epsilon_exploration: float):
        super().__init__()
        self.policy_nn = policy_nn
        self.n_packets_max = n_packets_max
        self.epsilon_exploration = epsilon_exploration

    def get_p_transmit(self, observation: Observation) -> float:
        with torch.no_grad():
            return self.policy_nn(format_actor_input([observation], self.n_packets_max)).item()

    def get_p_slot_and_slot_agnostic_p_transmission(self, observation: Observation):
        if not hasattr(self.policy_nn, "get_slot_and_transmission_probabilities"):
            raise ValueError("Actor architecture does not implement get_slot_and_transmission_probabilities method")
        with torch.no_grad():
            p_slot, slot_agnostic_p_transmission = self.policy_nn.get_slot_and_transmission_probabilities(
                format_actor_input([observation], self.n_packets_max)
            )
            return p_slot.item(), slot_agnostic_p_transmission.item()

    def action(self, observation):
        if observation.n_packets_buffer == 0:
            return 0, None
        with torch.no_grad():  # Gradients are only computed during replay
            p_transmit = self.get_p_transmit(observation)
            epsilon_factor = 1 - self.epsilon_exploration
            epsilon_per_action = self.epsilon_exploration / 2  # 2 possible actions
            action = np.random.choice([0, 1], p=[
                ((1 - p_transmit) * epsilon_factor) + epsilon_per_action,
                (p_transmit * epsilon_factor) + epsilon_per_action
            ])
        return action, None