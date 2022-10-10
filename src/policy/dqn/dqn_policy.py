import logging
import torch
import numpy as np

from src.data_classes import Observation
from src.policy.policy import Policy
from src.policy.dqn.utils import format_observation

logger = logging.getLogger(__name__)


class PolicyCommonDQN(Policy):
    def __init__(self, policy_dqn, epsilon_exploration: float):
        super().__init__()
        self.policy_dqn = policy_dqn
        self.epsilon_exploration = epsilon_exploration

    def _select_action(self, observation: Observation):
        if self.epsilon_exploration is not None:
            random_action = np.random.choice([False, True], p=[1 - self.epsilon_exploration, self.epsilon_exploration])
            if random_action:
                return np.random.choice([0, 1], p=[0.5, 0.5])
        return self.policy_dqn(format_observation(observation)).max(1).indices.item()  # Return scalar value

    def action(self, observation):
        if observation.n_packets_buffer == 0:
            return 0, None
        with torch.no_grad():
            action = self._select_action(observation)
        return action, None
