import torch
from typing import List

from settings import DEVICE
from src.data_classes import Observation, Transition


def format_actor_input(observations_batch: List[Observation], n_packets_max: int):
    return torch.tensor(
        [
            [
                obs.time_step,
                obs.n_packets_buffer / n_packets_max,
                obs.data_input,
                obs.ack
            ]
            for obs in observations_batch
        ],
        dtype=torch.float32,
        device=DEVICE
    ).view(
        len(observations_batch), -1
    )


def format_critic_input(transitions_batch: List[Transition], n_packets_max: int):
    return torch.tensor(
        [
            [
                [
                    obs.n_packets_buffer / n_packets_max,
                    obs.data_input,
                    obs.ack,
                    action
                ]
                for obs, action in zip(transition.agents_observations, transition.agents_actions)
            ]
            for transition in transitions_batch
        ],
        dtype=torch.float32,
        device=DEVICE
    ).view(
        len(transitions_batch), -1
    )
