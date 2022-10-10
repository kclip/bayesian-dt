import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, n_packets_max):
        super(DQN, self).__init__()
        input_size = (
                1 +  # Buffer
                1 +  # Data in
                2  # ACK (one hot)
        )
        output_size = 2  # [Q(input, a=0), Q(input, a=1)]

        hidden_size_1 = max(input_size // 2, 4)
        hidden_size_2 = max(hidden_size_1 // 2, 3)

        self.stack = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, output_size)
        )

    def forward(self, x):
        return self.stack(x)
