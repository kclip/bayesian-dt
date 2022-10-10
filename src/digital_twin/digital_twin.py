import logging
from copy import deepcopy
from typing import List, Optional
import numpy as np

from src.data_classes import Transition, RewardSettings
from src.digital_twin.environment_model import EnvironmentModel
from src.environment.environment import MACEnv
from src.environment.data_generator.transition import DataGeneratorDependencyInfo
from src.policy.policy import Policy
from src.policy.policy_optimizer import PolicyOptimizer
from src.policy.aloha.aloha_policy import PolicyAloha


logger = logging.getLogger(__name__)


class DigitalTwin(object):
    def __init__(self, policy_optimizer: PolicyOptimizer, **kwargs):
        self.policy_optimizer = policy_optimizer
        self.agents_policies = policy_optimizer.get_agents_policies()

    def train_model_step(self, step: int, transitions: List[Transition]) -> dict:
        raise NotImplementedError("Method 'train_model_step' must be implemented")

    def train_policy_step(self, step: int, transitions: List[Transition]) -> dict:
        raise NotImplementedError("Method 'train_policy_step' must be implemented")

    def init_virtual_env(self, env: Optional[MACEnv]) -> MACEnv:
        # Default: passthrough
        return env

    def update_virtual_env(self, step: int, virtual_env: MACEnv):
        # Inplace env update
        pass

    def post_training(self):
        # Restore policies to an exploitation mode (e.g. no epsilon-greedy exploration)
        self.agents_policies = self.policy_optimizer.get_agents_policies(training_policy=False)

    def get_agents_policies(self) -> List[Policy]:
        return self.agents_policies

    def get_info(self) -> dict:
        # Return info about current state to log
        return dict()

    def reset(self):
        # Reset policies internal state
        for policy in self.agents_policies:
            policy.reset()
        # Reset optimizer internal state (e.g. replay buffer)
        self.policy_optimizer.reset()

    def save_model(self, experiment_name: str):
        pass

    def load_model(self, experiment_name: str):
        pass


class DigitalTwinPolicyPassthrough(DigitalTwin):
    def train_model_step(self, step, transitions):
        raise ValueError("DigitalTwinPolicyPassthrough does not need to train model")

    def train_policy_step(self, step, transitions):
        # Update policies
        train_info = self.policy_optimizer.train_step(step, transitions)
        # Overwrite previous policies (new training policies might include exploratory behavior)
        self.agents_policies = self.policy_optimizer.get_agents_policies(training_policy=True)

        return train_info


class DigitalTwinModelBased(DigitalTwin):
    def __init__(
        self,
        policy_optimizer: PolicyOptimizer,
        n_agents: int,
        n_packets_max_agents: int,
        reward_settings: RewardSettings,
        data_generation_dependencies_info: List[DataGeneratorDependencyInfo],
        prior_dirichlet_concentration: float = 1,
        n_packets_rollouts: int = 1000,
        model_sampling_method: str = "posterior_sample",
        n_steps_between_model_update: int = 100
    ):
        super().__init__(policy_optimizer)
        self.n_agents = n_agents
        self.environment_model = EnvironmentModel(
            n_agents,
            n_packets_max_agents,
            data_generation_dependencies_info,
            prior_dirichlet_concentration=prior_dirichlet_concentration
        )
        self.reward_settings = reward_settings
        self.n_packets_rollouts = n_packets_rollouts
        self.model_sampling_method = model_sampling_method
        self.n_steps_between_model_update = n_steps_between_model_update

    def get_info(self):
        # Return posterior info
        return {
            "data_generation_posterior_params": self.environment_model.data_generation_posterior_params,
            "mpr_channel_posterior_params": self.environment_model.mpr_channel_posterior_params,
        }

    def init_virtual_env(self, env) -> MACEnv:
        return self.environment_model.init_env(
            self.n_packets_rollouts,
            self.reward_settings,
            self.model_sampling_method
        )

    def update_virtual_env(self, step, virtual_env):
        if step % self.n_steps_between_model_update == 0:
            # Inplace env update
            self.environment_model.update_env(virtual_env, self.model_sampling_method)
            # Print
            # print(self.model_sampling_method)
            # # print(f"Sampled data gen : \n{virtual_env.data_gen.transition._probabilities_maps}")
            # print(f"Sampled MPR : \n{virtual_env.channel.mpr_matrix}\n")

    def save_model(self, experiment_name: str):
        self.environment_model.save(experiment_name)

    def load_model(self, experiment_name: str, prior_dirichlet_concentration: float = None):
        # Load model
        self.environment_model = EnvironmentModel.load(experiment_name)

        # Update prior dirichlet concentration of the loaded model to the one specified
        # (useful when changing the prior in order to have well defined MAP estimates)
        if (
            (prior_dirichlet_concentration is not None) and
            (prior_dirichlet_concentration != self.environment_model.prior_dirichlet_concentration)
        ):
            logger.warning(
                f"Updating prior concentration in loaded environment model from "
                f"'{self.environment_model.prior_dirichlet_concentration}' to '{prior_dirichlet_concentration}' !"
            )
            self.environment_model.update_prior_dirichlet_concentration(prior_dirichlet_concentration)


class DigitalTwinConcurrentModel(DigitalTwinModelBased):
    def __init__(
        self,
        # Model based DT args
        policy_optimizer: PolicyOptimizer,
        n_agents: int,
        n_packets_max_agents: int,
        reward_settings: RewardSettings,
        data_generation_dependencies_info: List[DataGeneratorDependencyInfo],
        prior_dirichlet_concentration: float = 1,
        n_packets_rollouts: int = 1000,
        model_sampling_method: str = "posterior_sample",
        n_steps_between_model_update: int = 100,
        # Class specific args
        max_steps_rollouts: int = 100,
        min_steps_rollouts: int = 2000,
        frequency_rollouts: int = 500,
        n_episodes_rollouts: int = 3,
    ):
        super().__init__(
            policy_optimizer,
            n_agents,
            n_packets_max_agents,
            reward_settings,
            data_generation_dependencies_info,
            prior_dirichlet_concentration=prior_dirichlet_concentration,
            n_packets_rollouts=n_packets_rollouts,
            model_sampling_method=model_sampling_method,
            n_steps_between_model_update=n_steps_between_model_update
        )
        self.min_steps_rollouts = min_steps_rollouts
        self.frequency_rollouts = frequency_rollouts
        self.n_episodes_rollouts = n_episodes_rollouts
        self.max_steps_rollouts = max_steps_rollouts

    def train_policy_step(self, step, transitions):
        raise ValueError("DigitalTwinConcurrentModel implements policy optimization concurrently with  model learning")

    def train_model_step(self, step, transitions):
        train_info = dict()
        # Update models
        self.environment_model.update_model(transitions)
        # Rollout (update policies)
        if (self.min_steps_rollouts <= 0) and (step % self.frequency_rollouts == 0):
            train_info = self.virtual_rollout()
        # Overwrite previous policies (new training policies might include exploratory behavior)
        self.agents_policies = self.policy_optimizer.get_agents_policies(training_policy=True)
        # Update counter for rollouts start
        self.min_steps_rollouts = max(0, self.min_steps_rollouts - 1)

        return train_info

    def virtual_rollout(self) -> dict:
        virtual_rollout_info = dict()
        virtual_env = self.init_virtual_env(None)
        log_freq = self.max_steps_rollouts // 10
        for n_episode in range(self.n_episodes_rollouts):
            # Reset environment and internal states
            agents_observations = virtual_env.reset()
            self.policy_optimizer.reset()
            agents_policies = self.policy_optimizer.get_agents_policies(training_policy=True)
            agents_reward = [0] * self.n_agents
            step = 0

            # Play simulation
            while step < self.max_steps_rollouts:
                virtual_env.render()

                # Logging
                if step % log_freq == 0:
                    logger.debug(f"----- Step t = {str(step)} -----")
                    logger.debug(f"N packets left : {virtual_env.data_gen.n_packets_left}")
                    logger.debug(f"Data in t : {[obs.data_input for obs in agents_observations]}")
                    logger.debug(f"Buffers t : {[obs.n_packets_buffer for obs in agents_observations]}")

                # Agents actions at time step t
                agents_actions = [
                    agent.action(observation)[0]
                    for agent, observation in zip(agents_policies, agents_observations)
                ]

                # Transition to new state
                agents_next_observations, agents_rewards, done, info = virtual_env.step(
                    agents_actions,
                    cooperative_reward=self.reward_settings.cooperative_reward
                )
                transition = Transition(
                    agents_observations=agents_observations,
                    agents_actions=agents_actions,
                    agents_next_observations=agents_next_observations,
                    agents_rewards=agents_rewards,
                )

                # Train the policies
                self.policy_optimizer.train_step(step, [transition])

                # Sample new virtual environment dynamics from model
                self.environment_model.update_env(virtual_env, self.model_sampling_method)

                # Logging
                if step % log_freq == 0:
                    logger.debug(f"Actions t : {list(agents_actions)}")
                    logger.debug(f"ACK in t+1: {[int(obs.ack) for obs in agents_observations]}")
                    logger.debug(f"Reward per agent {sum(agents_reward) / len(agents_reward)}")

                # Update temp vars
                step += 1
                agents_observations = agents_next_observations

        virtual_env.close()

        return virtual_rollout_info


class ExplorationPolicyType(object):
    DEFAULT = "default"
    RANDOM = "random"


class DigitalTwinSeparateModel(DigitalTwinModelBased):
    def __init__(
        self,
        # Model based DT args
        policy_optimizer: PolicyOptimizer,
        n_agents: int,
        n_packets_max_agents: int,
        reward_settings: RewardSettings,
        data_generation_dependencies_info: List[DataGeneratorDependencyInfo],
        prior_dirichlet_concentration: float = 1,
        n_packets_rollouts: int = 1000,
        model_sampling_method: str = "posterior_sample",
        n_steps_between_model_update: int = 100,
        # Class specific args
        exploration_policy_type: str = "default"
    ):
        super().__init__(
            policy_optimizer,
            n_agents,
            n_packets_max_agents,
            reward_settings,
            data_generation_dependencies_info,
            prior_dirichlet_concentration=prior_dirichlet_concentration,
            n_packets_rollouts=n_packets_rollouts,
            model_sampling_method=model_sampling_method,
            n_steps_between_model_update=n_steps_between_model_update
        )
        self.exploration_policy_optimizer = deepcopy(self.policy_optimizer)
        self.exploration_policy_type = exploration_policy_type

    def _update_exploration_policy(self, transitions: List[Transition]):
        if self.exploration_policy_type == ExplorationPolicyType.DEFAULT:
            # Keep initial policy
            return
        elif self.exploration_policy_type == ExplorationPolicyType.RANDOM:
            # ALOHA policies with transmission probabilities sampled uniformly
            p_transmit = float(np.random.rand())
            self.agents_policies = [
                PolicyAloha(p_transmit)
                for _ in range(self.n_agents)
            ]
        else:
            raise ValueError(f"Unknown exploration policy type '{self.exploration_policy_type}'")

    def train_model_step(self, step, transitions):
        # Update models
        self.environment_model.update_model(transitions)
        # Update exploration policy
        self._update_exploration_policy(transitions)
        # Return empty train info
        return dict()

    def train_policy_step(self, step, transitions):
        # Update policies
        train_info = self.policy_optimizer.train_step(step, transitions)
        # Overwrite previous policies (new training policies might include exploratory behavior)
        self.agents_policies = self.policy_optimizer.get_agents_policies(training_policy=True)

        return train_info
