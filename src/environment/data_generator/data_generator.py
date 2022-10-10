from typing import List, Dict
from src.environment.data_generator.transition import DataGeneratorTransition


class DataGenerator(object):
    def __init__(self, n_agents, n_packets, transition: DataGeneratorTransition):
        self.n_agents = n_agents
        self.n_packets_init = n_packets
        self.n_packets_left = n_packets
        self.transition = transition
        self.data_input_state = [0] * self.n_agents

    def _get_nb_packets(self, state):
        return sum(int(digit) for digit in state)

    def step(self, forced_next_data_input_state: list = None):
        if self.n_packets_left <= 0:
            self.data_input_state = [0] * self.n_agents
        else:
            if forced_next_data_input_state is None:
                self.data_input_state = self.transition.next_state(self.data_input_state)
            else:
                self.data_input_state = forced_next_data_input_state
            self.n_packets_left -= self._get_nb_packets(self.data_input_state)  # WARNING: can be negative on last step!

    def update_transition(self, new_transition: DataGeneratorTransition):
        self.transition = new_transition

    def reset(self):
        self.data_input_state = [0] * self.n_agents
        self.n_packets_left = self.n_packets_init

    def get_transition_probabilities(
            self,
            generated_packets: List[int],
            next_generated_packets: List[int]
    ) -> Dict[str, float]:
        return self.transition.get_transition_probabilities(generated_packets, next_generated_packets)
