import torch
import torch.nn as nn

from settings import DEVICE


class QValueNN(nn.Module):
    def __init__(self, n_agents: int):
        super().__init__()

        self.local_action_obs_size = (
             1 +  # Buffer
             1 +  # Data in
             1 +  # ACK
             1  # Action
         )
        input_size = self.local_action_obs_size * n_agents
        output_size = 1  # [Q(state, action)]

        hidden_size_1 = input_size * 4
        hidden_size_2 = input_size * 4
        hidden_size_3 = input_size * 4

        self.stack = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, hidden_size_3),
            nn.ReLU(),
            nn.Linear(hidden_size_3, output_size)
        )

    def forward(self, selected_agent_idx, joint_state_actions):
        return self.stack(joint_state_actions)

    def get_q_values(self, selected_agent_idx, joint_state_actions):
        raw_input = torch.clone(joint_state_actions)
        idx_selected_action = self.local_action_obs_size * selected_agent_idx + 3

        raw_input[:, idx_selected_action] = 0
        q_values_action_0 = self.stack(raw_input)

        raw_input[:, idx_selected_action] = 1
        q_values_action_1 = self.stack(raw_input)

        return torch.cat([q_values_action_0, q_values_action_1], dim=1)


class DecomposedQValueNN(nn.Module):
    def __init__(self, n_agents: int):
        super().__init__()
        self.n_agents = n_agents

        self.local_input_size = (
             1 +  # Buffer
             1 +  # Data in
             1 +  # ACK
             1  # Action
         )
        hidden_local_size_1 = 32
        hidden_local_size_2 = 32
        self.local_output_size = 16
        global_hidden_size = 32
        global_output_size = 2  # 1 output per possible action

        self.other_agents_stack = nn.Sequential(
            nn.Linear(self.local_input_size, hidden_local_size_1),
            nn.ReLU(),
            nn.Linear(hidden_local_size_1, hidden_local_size_2),
            nn.ReLU(),
            nn.Linear(hidden_local_size_2, self.local_output_size),
            nn.ReLU(),
        )

        self.selected_agent_stack = nn.Sequential(
            nn.Linear(self.local_input_size, hidden_local_size_1),
            nn.ReLU(),
            nn.Linear(hidden_local_size_1, hidden_local_size_2),
            nn.ReLU(),
            nn.Linear(hidden_local_size_2, self.local_output_size),
            nn.ReLU(),
        )

        self.global_stack = nn.Sequential(
            nn.Linear(2 * self.local_output_size, global_hidden_size),
            nn.ReLU(),
            nn.Linear(global_hidden_size, global_output_size)
        )

    def _base_computation(self, selected_agent_idx, joint_state_actions):
        # Sum local outputs
        batch_size = joint_state_actions.size()[0]
        sum_other_agents_outputs = torch.zeros(batch_size, self.local_output_size, dtype=torch.float32).to(DEVICE)
        for agent_idx, local_input in enumerate(joint_state_actions.split(self.local_input_size, dim=1)):
            if agent_idx == selected_agent_idx:
                selected_agent_outputs = self.selected_agent_stack(local_input)
                selected_agent_actions = local_input[:, 3].type(torch.int64).view(-1, 1)
            else:
                sum_other_agents_outputs += self.other_agents_stack(local_input)

        # Global Q-values for each action of the selected agent
        q_values = self.global_stack(torch.cat([selected_agent_outputs, sum_other_agents_outputs], dim=1))

        return q_values, selected_agent_actions

    def forward(self, selected_agent_idx, joint_state_actions):
        q_values, selected_actions = self._base_computation(selected_agent_idx, joint_state_actions)
        return q_values.gather(1, selected_actions)

    def get_q_values(self, selected_agent_idx, joint_state_actions):
        q_values, _ = self._base_computation(selected_agent_idx, joint_state_actions)
        return q_values


def select_critic_architecture(class_name: str) -> nn.Module:
    _dict_architectures = {
        "QValueNN": QValueNN,
        "DecomposedQValueNN": DecomposedQValueNN,
    }
    return _dict_architectures[class_name]


class PolicyFeedforwardNN(nn.Module):
    def __init__(self, n_agents: int, frame_length: int):
        super().__init__()
        input_size = (
            1 +  # Buffer
            1 +  # Data in
            1    # ACK
        )

        hidden_size_1 = 32
        hidden_size_2 = 32
        output_size = 1  # [p(send | state)]

        self.stack = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.Tanh(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.Tanh(),
            # Removing bias in last layer to encourage a state dependent policy
            nn.Linear(hidden_size_2, output_size, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # First input is timestep, ignored here
        return self.stack(x[:, 1:])


class PolicySlottedNN(nn.Module):
    """Outputs independent probabilities of transmitting for each slot in the frame"""
    def __init__(self, n_agents: int, frame_length: int):
        if frame_length is None:
            raise ValueError("frame_length must be specified for PolicySlottedNN policy")

        super().__init__()
        input_size = (
            1 +  # Buffer
            1 +  # Data in
            1    # ACK
        )

        hidden_size_1 = 32
        hidden_size_2 = 32
        self.frame_length = frame_length

        self.stack = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.Tanh(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.Tanh(),
            nn.Linear(hidden_size_2, self.frame_length),
            nn.Sigmoid()
        )

    def forward(self, x):
        p_transmit_all_slots = self.stack(x[:, 1:])
        time_slot = x[:, 0].type(torch.int64).remainder(self.frame_length).view(-1, 1)
        p_transmit_slot = p_transmit_all_slots.gather(1, time_slot)
        return p_transmit_slot


class PolicySlottedLegacyNN(nn.Module):
    def __init__(self, n_agents: int, frame_length: int):
        if frame_length is None:
            raise ValueError("frame_length must be specified for PolicySlottedNN policy")

        super().__init__()
        input_size = (
            1 +  # Buffer
            1 +  # Data in
            1    # ACK
        )

        hidden_size_1 = 32
        hidden_size_2 = 32
        output_size = 1  # [p(send | state)]
        self.frame_length = frame_length

        self.common_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.Tanh(),
        )

        self.p_transmit_stack = nn.Sequential(
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.Tanh(),
            # Removing bias in last layer to encourage a state dependent policy
            nn.Linear(hidden_size_2, output_size, bias=False),
            nn.Sigmoid()
        )

        self.slot_stack = nn.Sequential(
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.Tanh(),
            nn.Linear(hidden_size_2, self.frame_length),
            nn.Softmax(dim=1)
        )

    def get_slot_and_transmission_probabilities(self, x):
        common_out = self.common_stack(x[:, 1:])

        slot_agnostic_p_transmission = self.p_transmit_stack(common_out)

        slot_softmax = self.slot_stack(common_out)
        time_slot = x[:, 0].type(torch.int64).remainder(self.frame_length).view(-1, 1)
        p_slot = slot_softmax.gather(1, time_slot)

        return p_slot, slot_agnostic_p_transmission

    def forward(self, x):
        p_slot, slot_agnostic_p_transmission = self.get_slot_and_transmission_probabilities(x)
        return p_slot * slot_agnostic_p_transmission


class PolicySlottedLegacyNNv2(nn.Module):
    """Same as PolicySlottedLegacyNN but without a separate path for slot_agnostic_p_transmission"""
    def __init__(self, n_agents: int, frame_length: int):
        if frame_length is None:
            raise ValueError("frame_length must be specified for PolicySlottedNN policy")

        super().__init__()
        input_size = (
            1 +  # Buffer
            1 +  # Data in
            1    # ACK
        )

        hidden_size_1 = 32
        hidden_size_2 = 32
        output_size = 1  # [p(send | state)]
        self.frame_length = frame_length

        self.stack = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.Tanh(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.Tanh(),
            nn.Linear(hidden_size_2, self.frame_length),
            nn.Softmax(dim=1)
        )

    def get_slot_and_transmission_probabilities(self, x):
        slot_softmax = self.stack(x[:, 1:])
        time_slot = x[:, 0].type(torch.int64).remainder(self.frame_length).view(-1, 1)
        p_slot = slot_softmax.gather(1, time_slot)
        # p_slot = torch.clamp(p_slot * self.frame_length, max=1)
        p_slot = 1 - nn.ReLU()(1 - (p_slot * self.frame_length))

        return (
            p_slot,
            torch.tensor([1] * x.size()[0], dtype=torch.float32).view(-1, 1).to(DEVICE)  # Placeholder for slot_agnostic_p_transmission
        )

    def forward(self, x):
        p_slot, _ = self.get_slot_and_transmission_probabilities(x)
        return p_slot


class Aloha(nn.Module):
    def __init__(self, n_agents: int, frame_length: int):
        super().__init__()

        self.p_transmit = torch.tensor(1 / n_agents, dtype=torch.float32).to(DEVICE)

    def forward(self, x):
        return self.p_transmit.repeat(x.size()[0]).view(-1, 1)


def select_actor_architecture(class_name: str) -> nn.Module:
    _dict_architectures = {
        "PolicyFeedforwardNN": PolicyFeedforwardNN,
        "PolicySlottedNN": PolicySlottedNN,
        "PolicySlottedLegacyNN": PolicySlottedLegacyNN,
        "PolicySlottedLegacyNNv2": PolicySlottedLegacyNNv2,
        "Aloha": Aloha,
    }
    return _dict_architectures[class_name]
