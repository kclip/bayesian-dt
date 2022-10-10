import os
import logging
import pickle
import torch
import torch.nn as nn
from typing import List

from src.data_classes import Transition, AgentTransition
from src.policy.policy_optimizer import PolicyOptimizer
from src.policy.epsilon_greedy import EpsilonSchedule
from src.policy.replay_buffer import ReplayBuffer
from src.policy.dqn.architecture import DQN
from src.policy.dqn.dqn_policy import PolicyCommonDQN
from src.policy.dqn.utils import format_observation


logger = logging.getLogger(__name__)


class PolicyOptimizerCommonDQN(PolicyOptimizer):
    def __init__(
            self,
            n_agents: int,
            n_packets_max: int,
            gamma=0.99,
            batch_size=128,
            target_dqn_update_step=50,
            epsilon_schedule=None,
            min_steps_before_update=100,
            max_size_buffer=10000
    ):
        super().__init__()
        self.n_agents = n_agents
        self.n_packets_max = n_packets_max
        self.replay_buffer = ReplayBuffer(max_size_buffer)
        self.policy_dqn = DQN(n_packets_max)
        self.target_dqn = DQN(n_packets_max)
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())  # Init target and policy DQN with same parameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_dqn_update_step = target_dqn_update_step
        self.epsilon_schedule = epsilon_schedule or EpsilonSchedule(0.5, 0.1, 0.99999)
        self.min_steps_before_update = min_steps_before_update
        self.optimizer = torch.optim.RMSprop(self.policy_dqn.parameters(), lr=0.001)

    def _update_policy_dqn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample and batch
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = AgentTransition(*zip(*transitions))  # transpose

        observation_batch = torch.cat(batch.observation)
        action_batch = torch.cat(batch.action)
        next_observation_batch = torch.cat(batch.next_observation)
        reward_batch = torch.cat(batch.reward)

        # Predicted value and target
        q_values_policy = self.policy_dqn(observation_batch).gather(1, action_batch)
        q_values_target = self.target_dqn(next_observation_batch).max(1).values.detach().view(-1, 1)
        target = reward_batch + self.gamma * q_values_target

        # Loss
        criterion = nn.MSELoss()
        loss = criterion(q_values_policy, target)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_dqn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def _update_target_dqn(self):
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

    def _transform_action(self, action):
        return torch.tensor([[action]], dtype=torch.int64)

    def _transform_reward(self, reward):
        return torch.tensor([[reward]], dtype=torch.int64)

    def get_agents_policies(self, training_policy=False):
        # Epsilon greedy exploration if training
        epsilon_exploration = None
        if training_policy:
            epsilon_exploration = self.epsilon_schedule.get_epsilon()  # This updates the epsilon schedule

        return [
            PolicyCommonDQN(self.policy_dqn, epsilon_exploration) for _ in range(self.n_agents)
        ]

    def train_step(self, step, transitions):
        info = dict()

        # Store single agent level transitions in replay buffer
        for transition in transitions:
            for observation, action, next_observation, rewards in zip(
                    transition.agents_observations,
                    transition.agents_actions,
                    transition.agents_next_observations,
                    transition.agents_rewards
            ):
                self.replay_buffer.append(
                    AgentTransition(
                        observation=format_observation(observation),
                        action=self._transform_action(action),
                        next_observation=format_observation(next_observation),
                        reward=self._transform_reward(rewards)
                    )
                )

        # Update policy DQN
        if self.min_steps_before_update > 0:
            self.min_steps_before_update -= 1
        else:
            self._update_policy_dqn()

        # Update target DQN
        if step % self.target_dqn_update_step == 0:
            self._update_target_dqn()

        return info

    def save(self, experiment_name: str):
        folder = self.policy_optimizer_folder(experiment_name)
        filepath = os.path.join(folder, "class_dict.pickle")
        with open(filepath, "wb") as file:
            pickle.dump(self.__dict__, file)

    @classmethod
    def load(cls, experiment_name: str):
        folder = cls.policy_optimizer_folder(experiment_name, create_if_not_exists=False)
        filepath = os.path.join(folder, "class_dict.pickle")
        with open(filepath, "rb") as file:
            class_dict = pickle.load(file)
        class_obj = cls(class_dict["n_agents"], class_dict["n_packets_max"])
        class_obj.__dict__.update(class_dict)
        return class_obj

    def reset(self):
        self.replay_buffer = ReplayBuffer(self.replay_buffer.max_size)
