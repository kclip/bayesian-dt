from typing import List, Union
import numpy as np


class MPRChannel(object):
    def __init__(self, n_agents: int, mpr_matrix: Union[List[List[float]], np.ndarray]):
        self.n_agents = n_agents
        self.mpr_matrix = np.array(mpr_matrix)
        self.n_delivered_packets_space = list(range(n_agents + 1))
        self.ack_state = np.zeros(n_agents, dtype=np.int64)

    def step(self, packets_sent_indicator):
        packets_sent_indicator = np.array(packets_sent_indicator)

        # Get message acknowledgements
        # Probability of k packets among n being successfully decoded
        n = np.sum(packets_sent_indicator)
        k = np.random.choice(
            self.n_delivered_packets_space,
            p=self.mpr_matrix[n, :]
        )
        # Choose randomly k packets among n
        self.ack_state = np.zeros(self.n_agents, dtype=np.int64)
        if k > 0:
            acknowledged_packets_indexes = np.random.choice(
                np.where(packets_sent_indicator == 1)[0],
                k,
                replace=False
            )
            self.ack_state[acknowledged_packets_indexes] = 1

    def update_mpr_matrix(self, mpr_matrix: np.ndarray):
        self.mpr_matrix = mpr_matrix

    def reset(self):
        self.ack_state = np.zeros(self.n_agents, dtype=np.int64)

    def get_transition_probability(self, n_packets_sent: int, n_packets_received: int) -> float:
        return self.mpr_matrix[n_packets_sent, n_packets_received]
