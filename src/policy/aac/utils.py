import torch


def format_observation(observation, n_packets_max):
    return torch.cat([
        # Normalize number of packets in buffer
        torch.tensor([observation.n_packets_buffer / n_packets_max], dtype=torch.float32),
        torch.tensor([observation.data_input], dtype=torch.int64),
        torch.tensor([observation.ack], dtype=torch.int64),
    ]).type(
        torch.FloatTensor
    ).view(1, -1)  # One entry
