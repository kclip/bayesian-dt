import os
import logging
import pickle
import torch
import torch.nn as nn
from typing import List

from src.data_classes import Transition
from src.policy.policy_optimizer import PolicyOptimizer
from src.policy.utils import select_optimizer
from src.policy.aac.architecture import select_critic_architecture, select_actor_architecture
from src.policy.aac.utils import format_observation
from src.policy.aac.aac_policy import PolicyCommonAAC


logger = logging.getLogger(__name__)


class PolicyOptimizerCommonAAC(PolicyOptimizer):
    def __init__(
            self,
            n_agents: int,
            n_packets_max: int,
            critic_architecture_name="ValueNN",
            actor_architecture_name="PolicyNN",
            optimizer_name="sgd",
            gamma=0.99,
            min_steps_before_update=10,
            learning_rate_actor=0.001,
            learning_rate_critic=0.001,
            critic_optimizer_kwargs: dict = None,
            actor_optimizer_kwargs: dict = None,
            steps_between_actor_updates=1,
            steps_between_critic_updates=1,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.n_packets_max = n_packets_max
        self.steps_between_actor_updates = steps_between_actor_updates
        self.steps_between_critic_updates = steps_between_critic_updates
        self.policy_nn = select_actor_architecture(actor_architecture_name)()
        self.value_nn = select_critic_architecture(critic_architecture_name)()
        self.gamma = gamma
        self.min_steps_before_update = min_steps_before_update
        self.last_transition: Transition = None
        _actor_optimizer_kwargs = actor_optimizer_kwargs or dict()
        self.policy_optimizer = select_optimizer(optimizer_name)(
            self.policy_nn.parameters(), lr=learning_rate_actor, **_actor_optimizer_kwargs
        )
        _critic_optimizer_kwargs = critic_optimizer_kwargs or dict()
        self.value_optimizer = select_optimizer(optimizer_name)(
            self.value_nn.parameters(), lr=learning_rate_critic, **_critic_optimizer_kwargs
        )

    def _batch_last_transitions(self):
        observation_batch = torch.cat([
            format_observation(observation, self.n_packets_max)
            for observation in self.last_transition.agents_observations
        ])
        action_batch = torch.cat([
            self._transform_action(action) for action in self.last_transition.agents_actions
        ])
        next_observation_batch = torch.cat([
            format_observation(next_observation, self.n_packets_max)
            for next_observation in self.last_transition.agents_next_observations
        ])
        reward_batch = torch.cat([
            self._transform_reward(reward) for reward in self.last_transition.agents_rewards
        ])

        return observation_batch, action_batch, next_observation_batch, reward_batch

    def _update_nn_actor(self):
        observation_batch, action_batch, next_observation_batch, reward_batch = self._batch_last_transitions()

        # Update policy network
        # ---------------------
        advantage = (
            reward_batch +
            self.gamma * self.value_nn(next_observation_batch).view(-1, 1) -
            self.value_nn(observation_batch).view(-1, 1)
        ).detach()
        # Selected action probability
        p_transmit = self.policy_nn(observation_batch)
        p_action = (action_batch * p_transmit) + ((1 - action_batch) * (1 - p_transmit))
        opposite_score_policy = -torch.log(p_action) * advantage  # Element-wise multiplication
        loss_policy = torch.sum(opposite_score_policy)
        self.policy_optimizer.zero_grad()
        loss_policy.backward()
        for param in self.policy_nn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy_optimizer.step()

    def _update_nn_critic(self):
        observation_batch, action_batch, next_observation_batch, reward_batch = self._batch_last_transitions()

        # Update value network
        # --------------------
        v_target = (reward_batch + self.gamma * self.value_nn(next_observation_batch).view(-1, 1)).detach()
        v_value = self.value_nn(observation_batch).view(-1, 1)
        criterion_val = nn.MSELoss()
        loss_val = criterion_val(v_value, v_target)
        self.value_optimizer.zero_grad()
        loss_val.backward()
        for param in self.value_nn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.value_optimizer.step()

    def _transform_action(self, action):
        return torch.tensor([[action]], dtype=torch.int64)

    def _transform_reward(self, reward):
        return torch.tensor([[reward]], dtype=torch.int64)

    def get_agents_policies(self, training_policy=False):
        return [
            PolicyCommonAAC(self.policy_nn, self.value_nn, self.n_packets_max) for _ in range(self.n_agents)
        ]

    def train_step(self, step, transitions):
        info = dict()

        # Store last transition
        self.last_transition = transitions[-1]

        # Update policy and value NN
        if self.min_steps_before_update > 0:
            self.min_steps_before_update -= 1
        else:
            if step % self.steps_between_critic_updates == 0:
                self._update_nn_critic()
            if step % self.steps_between_actor_updates == 0:
                self._update_nn_actor()

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
        self.last_transition = None
