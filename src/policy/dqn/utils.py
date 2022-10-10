import torch
import torch.nn as nn

from src.data_classes import Observation

def format_observation(observation: Observation):
    return torch.cat([
        torch.tensor([observation.n_packets_buffer], dtype=torch.int64),
        torch.tensor([observation.data_input], dtype=torch.int64),
        nn.functional.one_hot(
            torch.tensor(observation.ack, dtype=torch.int64),
            num_classes=2
        ),
    ]).type(
        torch.FloatTensor
    ).view(1, -1)  # One entry