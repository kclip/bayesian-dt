import logging
from typing import List
import numpy as np

from src.data_classes import Observation, Metadata, RewardSettings


logger = logging.getLogger(__name__)


def check_metadata_consistency(metadata: Metadata):
    # Check all agents have at least and only one reference in the data generation joint distribution
    count_agent_ref = np.zeros(metadata.env_metadata.n_agents)
    for agent_dependencies in metadata.env_metadata.data_generator_dependencies_kwargs:
        count_agent_ref[agent_dependencies["joint_agents"]] += 1
    mask_no_ref = (count_agent_ref == 0)
    mask_duplicate_ref = (count_agent_ref > 1)
    if mask_no_ref.any() or mask_duplicate_ref.any():
        agents = np.arange(metadata.env_metadata.n_agents)
        raise ValueError(
            f"Agents {list(agents[mask_no_ref])} are not referenced in the joint data generation process and agents "
            f"{list(agents[mask_duplicate_ref])} are referenced more than once"
        )

    # COMA policy must have cooperative reward
    if (
        (metadata.train_metadata.policy_optimizer_class == "PolicyOptimizerCOMA") and
        (not metadata.env_metadata.cooperative_reward)
    ):
        raise ValueError("COMA optimizer only works with cooperative rewards")


def check_step_consistency(agents_observations: List[Observation], agents_actions: List[int]):
    agents_buffers = np.array([obs.n_packets_buffer for obs in agents_observations])

    # Check that agent does not try to send a packet when the buffer is empty
    if (
        (np.array(agents_actions) == 1) &
        (agents_buffers == 0)
    ).any():
        logger.warning("An agent with an empty buffer is trying to send a packet")

    # Check for negative number of packets in buffer
    if (agents_buffers < 0).any():
        raise ValueError("An agent has a negative number of packets in its buffer")


def get_max_abs_reward(reward_settings: RewardSettings, agents_reward_weights: List[int], n_packets_max: int):
    abs_rewards = [
        abs(reward_settings.reward_default),
        abs(reward_settings.reward_ack),
        abs(reward_settings.reward_overflow),
        abs(reward_settings.reward_collision),
        abs(reward_settings.buffer_penalty_amplitude * n_packets_max),
    ]
    max_reward = max(abs_rewards)
    return max_reward * sum(agents_reward_weights)
