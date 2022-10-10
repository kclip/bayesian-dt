import numpy as np
from typing import List, Dict, Iterable, Tuple
from collections import namedtuple
from itertools import product
from copy import copy

from src.utils import decode_integers_tuple


DataGeneratorDependencyInfo = namedtuple(
    "DataGeneratorDependencyInfo",
    ["joint_agents", "adjacent_agents", "probabilities_map_name"]
)


class DataGenerationProbabilitiesMap(object):
    """
    Defines a mapping from the data generation state c of a subgroup of agents N_k at time t to a joint probability of
    packet generation g of a subgroup of agents M_k at time t+1
    i.e. P(g^{M_k}_{t+1} = g | g^{N_k}_t = c) = adjacent_values_to_probabilities_map(c)
    For logging reasons, the adjacent_values_to_probabilities_map is specified as a dictionary where
        - keys represent covariates (i.e. values of g^{N_k}_t) c = [c_1, ..., c_n] as a coma delimited string
          f"{c_1},...,{c_n}"
        - values are a dictionary representing probabilities of packet generation in M_k : keys are packet generation
          states g = [g_1, ..., g_m] as string f"{g_1},...,{g_m}" and values are probabilities p (must sum to 1)
    """
    def __init__(
            self,
            name: str,
            n_joint_agents: int,
            probabilities_map: Dict[str, Dict[str, float]] = None,
            default_joint_distribution: Dict[str, float] = None
    ):
        self.name = name
        self._default_key = "default"
        self._encoded_p_map = probabilities_map
        self.n_joint_agents = n_joint_agents

        # All joint values combinations
        self._all_joint_agents_values = list(product(*[[0, 1] for _ in range(n_joint_agents)]))

        # Map adjacent values to lists of joint agents distribution
        self._dict_p_map = dict()  # Dict output
        self._p_map = dict()  # List output
        if probabilities_map is not None:
            for encoded_adjacent_values, encoded_joint_distribution in probabilities_map.items():
                adjacent_values = decode_integers_tuple(encoded_adjacent_values)
                self._dict_p_map[adjacent_values] = {
                    decode_integers_tuple(encoded_joint_values): joint_data_gen_probability
                    for encoded_joint_values, joint_data_gen_probability in encoded_joint_distribution.items()
                }
                self._p_map[adjacent_values] = [
                    self._dict_p_map[adjacent_values][joint_values] for joint_values in self._all_joint_agents_values
                ]

        self._dict_default_joint_distribution = None
        self._default_joint_distribution = None
        if default_joint_distribution is not None:
            self._dict_default_joint_distribution = {
                decode_integers_tuple(encoded_joint_values): joint_data_gen_probability
                for encoded_joint_values, joint_data_gen_probability in default_joint_distribution.items()
            }
            self._default_joint_distribution = [
                self._dict_default_joint_distribution[joint_values] for joint_values in self._all_joint_agents_values
            ]

    def get_joint_data_generation_distribution(self, adjacent_values: Iterable) -> Tuple[list, list]:
        if not isinstance(adjacent_values, tuple):
            adjacent_values = tuple(adjacent_values)
        return (
            copy(self._all_joint_agents_values),
            copy(self._p_map.get(adjacent_values, self._default_joint_distribution))
        )

    def get_transition_probability(self, adjacent_values: Iterable, joint_values: Iterable) -> float:
        _adjacent_values = adjacent_values if isinstance(adjacent_values, tuple) else tuple(adjacent_values)
        _joint_values = joint_values if isinstance(joint_values, tuple) else tuple(joint_values)
        return self._dict_p_map.get(_adjacent_values,  self._dict_default_joint_distribution)[_joint_values]


class DataGeneratorTransition(object):
    """
    Defines a graph representing the probabilistic dependencies between packet generation from time step t to t+1.
    Each element in the joint_dependencies list is a tuple (M_k, N_k, f_name) where:
        - M_k is a list  of jointly dependent agents at time t+1
        - N_k is a list  of adjacent agents at time t
        - f_name is the name of probability map
    The argument probabilities_maps is a list of mappings from states N_k to joint generation probabilities in M_k.
    """

    def __init__(
            self,
            n_agents: int,
            agents_dependencies: List[DataGeneratorDependencyInfo],
            probabilities_maps: List[DataGenerationProbabilitiesMap]
    ):
        self._n_agents = n_agents
        self._agents_dependencies = agents_dependencies
        self._probabilities_maps = probabilities_maps
        self._indexed_probabilities_maps = {
            p_map.name: p_map
            for p_map in probabilities_maps
        }

    def get_dependencies_info(self) -> List[DataGeneratorDependencyInfo]:
        """
        Returns list of tuples containing the jointly dependent agents M_k at time step t+1, the agents on which they
        depend on N_k at time t (markovian data gen process) and the name of the distribution linking N_k to M_k
        (which can be used as an id of the data generation distribution to model)
        """
        return self._agents_dependencies

    def next_state(self, data_input: List[int]):
        _data_input = np.array(data_input)
        data_gen = np.zeros(self._n_agents, dtype=np.int64)
        for agents_dependency in self._agents_dependencies:
            p_map = self._indexed_probabilities_maps[agents_dependency.probabilities_map_name]
            joint_values, joint_values_probabilities = p_map.get_joint_data_generation_distribution(
                _data_input[agents_dependency.adjacent_agents]
            )
            idx_joint_value = np.random.choice(len(joint_values), p=joint_values_probabilities)
            data_gen[agents_dependency.joint_agents] = joint_values[idx_joint_value]

        return data_gen

    def get_transition_probabilities(
            self,
            generated_packets: List[int],
            next_generated_packets: List[int]
    ) -> Dict[str, float]:
        _generated_packets = np.array(generated_packets)
        _next_generated_packets = np.array(next_generated_packets)
        data_gen_probabilities_per_dependency = dict()
        for agents_dependency in self._agents_dependencies:
            adjacent_agents_packet_generation = _generated_packets[agents_dependency.adjacent_agents]
            joint_agents_packet_generation = _next_generated_packets[agents_dependency.joint_agents]
            p_map = self._indexed_probabilities_maps[agents_dependency.probabilities_map_name]
            prob_data_gen_dependency = p_map.get_transition_probability(
                adjacent_agents_packet_generation,
                joint_agents_packet_generation
            )
            data_gen_probabilities_per_dependency[agents_dependency.probabilities_map_name] = prob_data_gen_dependency

        return data_gen_probabilities_per_dependency
