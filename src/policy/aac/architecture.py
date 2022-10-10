import torch.nn as nn


class ValueNN(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = (
            1 +  # Buffer
            1 +  # Data in
            1  # ACK
        )
        output_size = 1  # [V(state)]

        hidden_size_1 = 4
        hidden_size_2 = 3

        self.stack = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, output_size)
        )

    def forward(self, x):
        return self.stack(x)


class ValueNNv2(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = (
            1 +  # Buffer
            1 +  # Data in
            1  # ACK
        )
        output_size = 1  # [V(state)]

        hidden_size_1 = 32
        hidden_size_2 = 32
        hidden_size_3 = 32

        self.stack = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, hidden_size_3),
            nn.ReLU(),
            nn.Linear(hidden_size_3, output_size)
        )

    def forward(self, x):
        return self.stack(x)


def select_critic_architecture(class_name: str) -> nn.Module:
    _dict_architectures = {
        "ValueNN": ValueNN,
        "ValueNNv2": ValueNNv2
    }
    return _dict_architectures[class_name]


class PolicyNN(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = (
            1 +  # Buffer
            1 +  # Data in
            1  # ACK
        )

        hidden_size = 4
        output_size = 1  # [p(send | state)]

        self.stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.stack(x)


class PolicyNNv2(nn.Module):
    def __init__(self, n_agents: int):
        super().__init__()
        input_size = (
            1 +  # Buffer
            1 +  # Data in
            1  # ACK
        )

        hidden_size_1 = 32
        hidden_size_2 = 32
        output_size = 1  # [p(send | state)]

        self.stack = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.Tanh(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.Tanh(),
            nn.Linear(hidden_size_2, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.stack(x)


def select_actor_architecture(class_name: str) -> nn.Module:
    _dict_architectures = {
        "PolicyNN": PolicyNN,
        "PolicyNNv2": PolicyNNv2
    }
    return _dict_architectures[class_name]
