import os
import logging
import pickle
import torch
import numpy as np
from typing import List

from settings import DEVICE
from src.data_classes import Transition
from src.policy.policy_optimizer import PolicyOptimizer
from src.policy.buffer import NStepsBuffer
from src.policy.epsilon_greedy import EpsilonSchedule
from src.policy.utils import select_optimizer, extract_gradients_info
from src.policy.coma.architecture import select_critic_architecture, select_actor_architecture
from src.policy.coma.utils import format_actor_input, format_critic_input
from src.policy.coma.coma_policy import PolicyCOMA


logger = logging.getLogger(__name__)

GRADIENT_CLIPPING = 1


class PolicyOptimizerCOMA(PolicyOptimizer):
    def __init__(
            self,
            n_agents: int,
            n_packets_max: int,
            max_abs_reward: float,
            return_discount: float = 0.99,
            transitions_buffer_min_size: int = None,
            transitions_buffer_max_size: int = None,
            batch_size: int = 32,
            lambda_return_decay: float = 0.9,
            lambda_return_n_steps: int = 10,
            frame_length: int = None,
            # Critic
            critic_optimizer_name: str = "sgd",
            critic_architecture_name: str = "QValueNN",
            learning_rate_critic: float = 0.001,
            critic_optimizer_kwargs: dict = None,
            n_updates_between_critic_target_updates: int = 10,
            # Actor
            actor_optimizer_name: str = "sgd",
            actor_architecture_name: str = "PolicyFeedforwardNN",
            share_actor_network: bool = True,
            learning_rate_actor: float = 0.001,
            actor_optimizer_kwargs: dict = None,
            n_critic_updates_between_actor_updates: int = 1,
            # Entropy exploration
            entropy_exploration_n_steps: int = 0,
            entropy_exploration_temperature: float = 0,
            # Epsilon exploration
            epsilon_exploration_start: float = 0,
            epsilon_exploration_end: float = 0,
            epsilon_exploration_step: float = 0.99999,
            epsilon_exploration_update_frequency: int = 10
    ):
        super().__init__()
        self.n_agents = n_agents
        self.n_packets_max = n_packets_max
        self.max_abs_reward = max_abs_reward
        self.n_critic_updates_between_actor_updates = n_critic_updates_between_actor_updates
        self.actor_update_counter = -1
        self.n_updates_between_critic_target_updates = n_updates_between_critic_target_updates
        self.target_update_counter = -1
        self.batch_size = batch_size
        self.return_discount = return_discount
        self.lambda_return_decay = lambda_return_decay
        self.lambda_return_n_steps = lambda_return_n_steps
        self.frame_length = frame_length

        # Centralized critic
        max_abs_reward_with_exploration = (  # Maximum entropy exploration adds to the original reward
            max_abs_reward + entropy_exploration_temperature
        )
        self.return_scaling = self._get_return_scaling(  # Scale Q NN output to match target (helps converge faster)
            max_abs_reward_with_exploration,
            return_discount,
            lambda_return_n_steps,
            lambda_return_decay
        )
        critic_args = [n_agents]
        self.q_value_nn = select_critic_architecture(critic_architecture_name)(*critic_args).to(DEVICE)
        self.q_value_nn.eval()
        self.q_value_target_nn = select_critic_architecture(critic_architecture_name)(*critic_args).to(DEVICE)
        self.q_value_target_nn.load_state_dict(self.q_value_nn.state_dict())
        self.critic_optimizer_kwargs = {
            **(critic_optimizer_kwargs or dict()),
            "lr": learning_rate_critic
        }
        self.critic_optimizer_class = select_optimizer(critic_optimizer_name)
        self.q_value_optimizer = self._init_critic_optimizer()

        # Decentralized actors
        self.share_actor_network = share_actor_network
        self.optimize_actor = actor_architecture_name != "Aloha"
        actor_args = [n_agents, self.frame_length]
        if self.share_actor_network:
            _policy_nn = select_actor_architecture(actor_architecture_name)(*actor_args).to(DEVICE)
            self.policies_nn = [_policy_nn for _ in range(n_agents)]
        else:
            self.policies_nn = [
                select_actor_architecture(actor_architecture_name)(*actor_args).to(DEVICE)
                for _ in range(n_agents)
            ]
        for policy_nn in self.policies_nn:
            policy_nn.eval()
        self.actor_optimizer_kwargs = {
            **(actor_optimizer_kwargs or dict()),
            "lr": learning_rate_actor
        }
        self.policy_optimizer = None
        self.actor_optimizer_class = select_optimizer(actor_optimizer_name)
        if self.optimize_actor:
            self.policy_optimizer = self._init_actor_optimizer()

        # Experience buffer
        # Min size of stored transitions for critic updates
        _transitions_buffer_min_size = transitions_buffer_min_size or (lambda_return_n_steps + batch_size)
        if _transitions_buffer_min_size <= lambda_return_n_steps:
            raise ValueError("Cannot compute n-step return with less than n+1 samples in buffer")
        self.transitions_buffer_min_size = _transitions_buffer_min_size
        # Max buffer size
        _transitions_buffer_max_size = transitions_buffer_max_size or (lambda_return_n_steps + batch_size)
        if _transitions_buffer_max_size < _transitions_buffer_min_size:
            raise ValueError("Buffer max size should be greater or equal to min size")
        self.transitions_buffer_max_size = _transitions_buffer_max_size
        self.transitions_buffer = NStepsBuffer(self.transitions_buffer_max_size)

        # Entropy exploration
        self.entropy_exploration_n_steps = entropy_exploration_n_steps
        self.entropy_exploration_temperature = entropy_exploration_temperature

        # Epsilon exploration
        # Note: this is mostly useful to explore enough states to learn the Q-value while using an Aloha actor
        self.epsilon_scheduler = EpsilonSchedule(
            epsilon_exploration_start, epsilon_exploration_end, epsilon_exploration_step
        )
        self.current_epsilon_exploration = epsilon_exploration_start
        self.epsilon_exploration_update_frequency = epsilon_exploration_update_frequency

    def _init_critic_optimizer(self):
        return self.critic_optimizer_class(
            self.q_value_nn.parameters(), **self.critic_optimizer_kwargs
        )

    def _init_actor_optimizer(self):
        if self.share_actor_network:
            policies_paramters = self.policies_nn[0].parameters()
        else:
            policies_paramters = []
            for policy_nn in self.policies_nn:
                policies_paramters += list(policy_nn.parameters())
        return self.actor_optimizer_class(  # All policies share a common optimizer
            policies_paramters, **self.actor_optimizer_kwargs
        )

    def _get_return_scaling(self, max_abs_reward, return_discount, lambda_return_n_steps, lambda_return_decay):
        if (lambda_return_decay == 1) or (lambda_return_n_steps == 1):
            # max n-step return
            return max_abs_reward * (
                (1 - (return_discount ** (lambda_return_n_steps + 1))) /
                (1 - return_discount)
            )
        else:
            # max n-step truncated lambda return
            return (max_abs_reward / (1 - return_discount)) * (
                1 - (
                    (return_discount ** (lambda_return_n_steps + 1)) *
                    (lambda_return_decay ** (lambda_return_n_steps - 1))
                ) - (
                    (return_discount ** 2) * (
                        (1 - lambda_return_decay) /
                        (1 - lambda_return_decay * return_discount)
                    ) * (
                        1 - (
                            (return_discount ** (lambda_return_n_steps + 1)) *
                            (lambda_return_decay ** (lambda_return_n_steps - 1))
                        )
                    )
                )
            )

    def _update_nn_actors(self):
        # Get <batch_size> transitions
        actor_batch_size = self.batch_size
        batch_indexes = np.random.randint(
            0,
            len(self.transitions_buffer.buffer),
            size=actor_batch_size
        )
        transitions = [
            self.transitions_buffer.buffer_values[i]
            for i in batch_indexes
        ]

        # Format inputs
        formatted_input_critic = format_critic_input(
            transitions, self.n_packets_max
        )
        formatted_input_actors = [
            format_actor_input(
                [
                    transition.agents_observations[agent_idx]
                    for transition in transitions
                ],
                self.n_packets_max
            )
            for agent_idx in range(self.n_agents)
        ]
        formatted_actions = torch.tensor(
            [transition.agents_actions for transition in transitions],
            dtype=torch.int64
        ).to(DEVICE)

        # Compute COMA advantage
        counterfactual_advantages = torch.zeros(self.batch_size, self.n_agents, dtype=torch.float32).to(DEVICE)  # Size [batch ; n_agents]
        with torch.no_grad():
            for agent_idx in range(self.n_agents):
                all_q_values = self.q_value_nn.get_q_values(agent_idx, formatted_input_critic)  # Size [batch;n_actions]
                p_actions = self.policies_nn[agent_idx](formatted_input_actors[agent_idx])
                actions = formatted_actions[:, agent_idx].view(-1, 1)
                q_value = all_q_values.gather(1, actions)
                counterfactual_advantages[:, agent_idx] = (
                    q_value -
                    ((1 - p_actions) * all_q_values[:, 0].view(-1, 1)) -
                    (p_actions * all_q_values[:, 1].view(-1, 1))
                ).view(-1)

        # Get policy network action probabilities for sampled transitions
        for policy_nn in self.policies_nn:
            policy_nn.train()
        transmission_probabilities = torch.cat([  # Size [batch ; agent]
            policy_nn(local_input)
            for agent_idx, (policy_nn, local_input) in enumerate(zip(self.policies_nn, formatted_input_actors))
        ], dim=1)
        action_probabilities = (
            (formatted_actions * transmission_probabilities) +
            ((1 - formatted_actions) * (1 - transmission_probabilities))
        )

        # Compute loss as opposite of actor/critic score
        opposite_policies_scores = -torch.log(action_probabilities) * counterfactual_advantages

        # Update policies networks
        loss_policy = torch.mean(opposite_policies_scores)
        self.policy_optimizer.zero_grad()
        loss_policy.backward()
        for policy_nn in self.policies_nn:
            for param in policy_nn.parameters():
                param.grad.data.clamp_(-GRADIENT_CLIPPING, GRADIENT_CLIPPING)
        self.policy_optimizer.step()

        # Set network to eval to pursue simulation
        for policy_nn in self.policies_nn:
            policy_nn.eval()

        # Get gradients info
        gradients_info = [
            extract_gradients_info(policy_nn.named_parameters())
            for policy_nn in self.policies_nn
        ]

        return {
            "loss_actor": loss_policy.detach().item(),
            "gradients_info_actors": gradients_info
        }

    def _truncated_lambda_return(
            self,
            batch_steps: torch.Tensor,
            batch_rewards: torch.Tensor,
            selected_agent_idx: int
    ) -> torch.Tensor:
        # Compute Q-values estimates for future steps
        with torch.no_grad():
            inputs_len = batch_steps.size()[2]
            next_inputs = batch_steps[:, 1:, :].reshape(self.batch_size * self.lambda_return_n_steps, inputs_len)

            next_step_q_values = self.q_value_target_nn(
                selected_agent_idx, next_inputs
            ).view(
                self.batch_size,
                self.lambda_return_n_steps
            )

        # Compute and store k-steps returns for k in [1,n]
        k_steps_reward = torch.zeros((self.batch_size, 1), dtype=torch.float32).to(DEVICE)
        k_steps_returns = []
        discount_k = 1
        for k in range(self.lambda_return_n_steps):
            k_steps_reward += discount_k * batch_rewards[:, k].view(self.batch_size, 1)
            k_steps_returns.append(
                k_steps_reward +
                discount_k * self.return_discount * next_step_q_values[:, k].view(self.batch_size, 1)
            )
            discount_k *= self.return_discount

        # Compute lambda weighted returns (truncated at step n)
        if self.lambda_return_decay == 1:  # truncated lambda=1 return equals n-step return
            return k_steps_returns[self.lambda_return_n_steps-1]
        elif self.lambda_return_n_steps == 0:  # truncated lambda=0 return equals 1-step return (TD(0))
            return k_steps_returns[0]
        else:  # general case
            lambda_returns = torch.zeros((self.batch_size, 1), dtype=torch.float32).to(DEVICE)
            decay_k = 1
            for k in range(self.lambda_return_n_steps - 1):
                lambda_returns += decay_k * k_steps_returns[k]
                decay_k *= self.lambda_return_decay
            lambda_returns = (
                (1 - self.lambda_return_decay) * lambda_returns +
                decay_k * k_steps_returns[self.lambda_return_n_steps-1]
            )
            return lambda_returns

    def _update_nn_critic(self):
        # Sample <self.batch_size> steps from where to start the n-step estimate of the return
        batch_indexes = np.random.randint(
            0,
            len(self.transitions_buffer.buffer) - (self.lambda_return_n_steps + 1),
            size=self.batch_size
        )

        # Extract all transitions needed
        all_transitions = []
        buffer_values = self.transitions_buffer.buffer_values
        for buffer_start_idx in batch_indexes:
            # add "+ 1" for the Q-value estimate at the n+1 step ahead
            all_transitions += buffer_values[buffer_start_idx:buffer_start_idx + self.lambda_return_n_steps + 1]

        # Formated inputs for target network
        batch_steps = format_critic_input(
            all_transitions, self.n_packets_max
        ).view(
            self.batch_size,
            self.lambda_return_n_steps + 1,
            -1  # Input size
        )

        # Get all rewards
        all_rewards = torch.tensor(
            [
                transition.agents_rewards[0]  # Agents must share the same reward
                for transition in all_transitions
            ],
            dtype=torch.float32
        ).to(DEVICE)

        # Entropy exploration (exploration by rewarding non-deterministic policies)
        if (
            (self.entropy_exploration_temperature != 0) and
            (self.entropy_exploration_n_steps > 0)
        ):
            # Get sampled actions log probability
            formatted_input_actors = [
                format_actor_input(
                    [
                        transition.agents_observations[agent_idx]
                        for transition in all_transitions
                    ],
                    self.n_packets_max
                )
                for agent_idx in range(self.n_agents)
            ]
            formatted_actions = torch.tensor(
                [transition.agents_actions for transition in all_transitions],
                dtype=torch.float32
            ).to(DEVICE)
            actors_entropies = torch.zeros(len(all_transitions), self.n_agents, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                for agent_idx in range(self.n_agents):
                    p_transmit = self.policies_nn[agent_idx](formatted_input_actors[agent_idx])
                    log_loss = (
                            (formatted_actions[:, agent_idx].view(-1, 1) * (-torch.log(p_transmit))) +
                            ((1 - formatted_actions[:, agent_idx]).view(-1, 1) * (-torch.log(1 - p_transmit)))
                    )
                    actors_entropies[:, agent_idx] = log_loss.view(-1)
                actors_avg_entropy = actors_entropies.mean(dim=1).view(-1, 1)

            # Update reward with average policies entropies
            all_rewards = all_rewards + self.entropy_exploration_temperature * actors_avg_entropy.view(-1)

            # Update steps counter to deactivate entropy exploration
            self.entropy_exploration_n_steps -= 1
            if self.entropy_exploration_n_steps == 0:
                logger.warning("Deactivating entropy exploration...")
                # Update return scaling (max reward changes)
                self.return_scaling = self._get_return_scaling(
                    self.max_abs_reward,
                    self.return_discount,
                    self.lambda_return_n_steps,
                    self.lambda_return_decay
                )

        # Group rewards per n-steps
        batch_rewards = all_rewards.view(
            self.batch_size,
            self.lambda_return_n_steps + 1,
        )

        # Select agent for Q-network outputs
        selected_agent_idx = np.random.randint(0, self.n_agents)

        # Compute <batch_size> estimated returns
        estimated_returns = self._truncated_lambda_return(batch_steps, batch_rewards, selected_agent_idx)
        # Normalize return (helps Q network convergence). WARNING: plotting the Q-value won't represent the true return
        estimated_returns = estimated_returns / self.return_scaling

        # Compute estimated values
        self.q_value_nn.train()
        batch_inputs = batch_steps[:, 0, :]

        estimated_q_values = self.q_value_nn(selected_agent_idx, batch_inputs).view(-1, 1)

        # Gradient descent step
        loss_val = torch.mean((estimated_returns - estimated_q_values)**2)
        self.q_value_optimizer.zero_grad()
        loss_val.backward()
        for param in self.q_value_nn.parameters():
            param.grad.data.clamp_(-GRADIENT_CLIPPING, GRADIENT_CLIPPING)
        self.q_value_optimizer.step()

        # Set network to eval to pursue simulation
        self.q_value_nn.eval()

        # Get gradients info
        gradients_info = extract_gradients_info(self.q_value_nn.named_parameters())

        return {
            "loss_critic": loss_val.detach().item(),
            "estimated_target_critic": torch.mean(estimated_returns.detach()).item(),
            "estimated_q_value_critic": torch.mean(estimated_q_values.detach()).item(),
            "gradients_info_critic": gradients_info,
        }

    def get_agents_policies(self, training_policy=False):
        epsilon_exploration = self.current_epsilon_exploration if training_policy else 0
        return [
            PolicyCOMA(policy_nn, self.n_packets_max, epsilon_exploration)
            for policy_nn in self.policies_nn
        ]

    def get_value_critic(self, transition: Transition) -> float:
        with torch.no_grad():
            return self.q_value_nn(0, format_critic_input([transition], self.n_packets_max)).item()

    def train_step(self, step: int, transitions: List[Transition]):
        # Return train info
        info = dict()

        # Store last transition
        self.transitions_buffer.append(transitions)

        # If buffer is full enough, perform gradient updates
        if len(self.transitions_buffer) >= self.transitions_buffer_min_size:
            # Update critic
            info_critic = self._update_nn_critic()
            info.update(info_critic)
            self.target_update_counter = (self.target_update_counter + 1) % self.n_updates_between_critic_target_updates
            self.actor_update_counter = (self.actor_update_counter + 1) % self.n_critic_updates_between_actor_updates

            # Update actors
            if self.optimize_actor and (self.actor_update_counter == 0):
                info_actor = self._update_nn_actors()
                info.update(info_actor)
                # Critic learning is on-policy, we thus must delete the stored experience
                self.transitions_buffer.reset()

            # Update critic target
            if self.target_update_counter == 0:
                self.q_value_target_nn.load_state_dict(self.q_value_nn.state_dict())

        # Update epsilon exploration
        if step % self.epsilon_exploration_update_frequency == 0:
            self.current_epsilon_exploration = self.epsilon_scheduler.get_epsilon(update_epsilon=True)

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
        class_obj = cls(class_dict["n_agents"], class_dict["n_packets_max"], class_dict["max_abs_reward"])
        class_obj.__dict__.update(class_dict)
        return class_obj

    def reset(self):
        self.target_update_counter = -1
        self.actor_update_counter = -1
        self.transitions_buffer.reset()
        self.q_value_optimizer = self._init_critic_optimizer()
        if self.optimize_actor:
            self.policy_optimizer = self._init_actor_optimizer()
        self.q_value_nn.eval()
        for policy_nn in self.policies_nn:
            policy_nn.eval()
