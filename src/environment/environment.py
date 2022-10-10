from typing import List, Dict

import numpy as np

from src.data_classes import Observation, Step, Transition
from src.environment.agent.agent import Agent
from src.environment.channel.channel import MPRChannel
from src.environment.data_generator.data_generator import DataGenerator
from src.environment.data_generator.transition import DataGeneratorTransition


class MACEnv(object):
    def __init__(
            self,
            n_agents: int,
            data_gen: DataGenerator,
            channel: MPRChannel,
            agents: List[Agent]
    ):
        """
        Environment class.
        Observation space topology (the "spaces" are taken from the "gym" library specification):
            spaces.Tuple([
                spaces.MultiDiscrete([
                    agent.n_packets_max + 1,  # Number of packets in agent's buffer
                    2,  # Data input: 0 -> no packet generated during time slot, 1 -> Packet generated
                    2,  # ACK: 0 ->  no packet sent at previous time slot or not received,
                        #      1 -> packet sent at previous time slot successfully received
                    2  # Channel state: 0 -> Good state, 1 -> Bad state
                ])
                for agent in agents
            ])
        Action space topology:
            spaces.Tuple([
                spaces.Discrete(2)
                for _ in agents
            ])
        """
        self.n_agents = n_agents
        self.data_gen = data_gen
        self.channel = channel
        self.agents = agents
        self.time_step = 0

    def _zip_observations(self, agents, agents_data_input, agents_ack, time_step):
        return tuple(
            Observation(
                n_packets_max=agent.n_packets_max,
                n_packets_buffer=agent.n_packets_buffer,
                data_input=data_input,
                ack=ack,
                time_step=time_step
            )
            for agent, data_input, ack in zip(
                agents,
                agents_data_input,
                agents_ack
            )
        )

    def is_terminal_state(self):
        return (
            (self.data_gen.n_packets_left <= 0)  # No packets left to send
            and (sum(agent.n_packets_buffer for agent in self.agents) == 0)  # All agents have sent all their packets
        )

    def step(self, agents_actions, forced_episode_next_step: Step = None, cooperative_reward: bool = False):
        # Note: this step starts from the moment an action is taken at time t and ends just before the action at time
        # (t+1).
        # Thus we have to compute these operations in the following order:
        #     - Transform agent action into packets being actually sent (only possible if agent buffer is not empty)
        #     - Update acknowledgement signal to timestep (t+1) (depends on MPR matrix at t) (ACK_t -> ACK_{t+1})
        #     - Update the data input (d_t -> d_{t+1})
        #     - Update MPR matrix to timestep (t+1)
        #     - Update the buffer to timestep (t+1) (depends on ACK and data input at (t+1)) (B_t -> B_{t+1})
        #     - Compute the reward

        # Info object (contextual data not present in state, useful for metrics)
        info = {
            "buffer_overflow": [],
            "channel_collision": []
        }

        # An agent trying to send a packet with an empty buffer defaults to no packet being sent
        agents_buffers = [agent.n_packets_buffer for agent in self.agents]
        agents_actions = np.where(
            (np.array(agents_actions) == 1) & (np.array(agents_buffers) > 0),
            1,  # if Action = "send" and buffer not empty
            0  # otherwise
        )

        # Update ACK
        self.channel.step(agents_actions)

        # Update data input
        if forced_episode_next_step is None:
            self.data_gen.step()
        else:
            self.data_gen.step(forced_next_data_input_state=forced_episode_next_step.state.data_generated)

        # Update buffers and compute rewards
        agents_rewards = []
        for agent, next_data_input, next_ack, action in zip(
                self.agents,
                self.data_gen.data_input_state,
                self.channel.ack_state,
                agents_actions
        ):
            reward, agent_info = agent.step(next_data_input, next_ack, action)
            agents_rewards.append(reward)
            info["buffer_overflow"].append(agent_info["buffer_overflow"])
            info["channel_collision"].append(agent_info["channel_collision"])

        # Update time step
        self.time_step += 1

        # New state
        agents_observations = self._zip_observations(
            self.agents,
            self.data_gen.data_input_state,
            self.channel.ack_state,
            self.time_step
        )

        # Check for terminal state
        done = self.is_terminal_state()

        # Cooperative reward
        if cooperative_reward:
            agents_rewards = [np.sum(agents_rewards)] * self.n_agents

        return agents_observations, agents_rewards, done, info

    def render(self):
        pass

    def update_dynamics(self, data_generator_transition: DataGeneratorTransition, mpr_channel_matrix: np.ndarray):
        self.data_gen.update_transition(data_generator_transition)
        self.channel.update_mpr_matrix(mpr_channel_matrix)

    def reset(self):
        self.data_gen.reset()
        self.channel.reset()
        for agent in self.agents:
            agent.reset()
        self.time_step = 0

        return self._zip_observations(
            self.agents,
            self.data_gen.data_input_state,
            self.channel.ack_state,
            self.time_step
        )

    def close(self):
        pass

    def get_transition_probabilities(self, transition: Transition) -> Dict[str, float]:
        # Data generation
        generated_packets = [observation.data_input for observation in transition.agents_observations]
        next_generated_packets = [observation.data_input for observation in transition.agents_next_observations]
        data_gen_probabilities_per_dependency = self.data_gen.get_transition_probabilities(
            generated_packets,
            next_generated_packets
        )

        # MPR matrix
        n_packets_sent = sum(transition.agents_actions)
        n_packets_received = sum(observation.ack for observation in transition.agents_next_observations)
        prob_channel = self.channel.get_transition_probability(n_packets_sent, n_packets_received)

        return {
            **{
                f"data_gen_{k}": v
                for k, v in data_gen_probabilities_per_dependency.items()
            },
            "mpr_channel": prob_channel
        }
