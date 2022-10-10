import os
import pickle
import logging
from typing import List
from itertools import product
from collections import Counter
from scipy.stats import dirichlet
import numpy as np

from settings import DATA_FOLDER, ENVIRONMENT_MODEL_FOLDER
from src.data_classes import Transition, RewardSettings
from src.utils import encode_integers_tuple, max_likelihood_estimate, max_a_posteriori_estimate
from src.environment.data_generator.transition import DataGeneratorDependencyInfo, DataGenerationProbabilitiesMap, \
    DataGeneratorTransition
from src.environment.channel.channel import MPRChannel
from src.environment.data_generator.data_generator import DataGenerator
from src.environment.agent.agent import Agent
from src.environment.environment import MACEnv


logger = logging.getLogger(__name__)

NAN_SAMPLE_RETRY = 10


class SampleMethod(object):
    POSTERIOR_SAMPLE = "posterior_sample"
    MAXIMUM_LIKELIHOOD = "maximum_likelihood"
    MAXIMUM_A_POSTERIORI = "maximum_a_posteriori"


class EnvironmentModel(object):
    def __init__(
        self,
        n_agents: int,
        n_packets_max_agents: int,
        data_generation_dependencies_info: List[DataGeneratorDependencyInfo],
        prior_dirichlet_concentration: float = 1
    ):
        self.n_agents = n_agents
        self.n_packets_max_agents = n_packets_max_agents
        self.prior_dirichlet_concentration = prior_dirichlet_concentration

        # Dirichlet priors for data generation
        self._data_generation_dependencies_info = data_generation_dependencies_info
        self._data_generation_map_names_to_dirichlet_covariate_length = {
            p_map_name: {
                "n_joint_agents": n_covariates[0],
                "n_adjacent_agents": n_covariates[1]
            }
            for p_map_name, n_covariates in {
                # Generate set of tuples to avoid repetition
                (
                        dependency_info.probabilities_map_name,
                        (
                            len(dependency_info.joint_agents),
                            len(dependency_info.adjacent_agents)
                        )
                )
                for dependency_info in self._data_generation_dependencies_info
            }
        }
        self._data_generation_transitions_counters = {
            # Count data generation transition occurrences for each different data generation probabilities map
            probabilities_map_name: Counter({
                # Number of transition occurrences for a given data generation probability map
                #     - key: dirichlet covariates (g^{M_k}_{t+1}, g^{N_k}_t) where g^{M_k}_{t+1} are the values of the
                #            joint agents at time t+1, and g^{N_k}_t the values of the adjacent agents at time t
                #     - value: nb of occurences (dirichlet parameters)
                joint_and_adjacent_agents_values: self.prior_dirichlet_concentration
                for joint_and_adjacent_agents_values in product(*[
                    [0, 1] for _ in range(n_covariates["n_joint_agents"] + n_covariates["n_adjacent_agents"])
                ])
            })
            for probabilities_map_name, n_covariates in
            self._data_generation_map_names_to_dirichlet_covariate_length.items()
        }

        # Dirichlet priors for MPR channel
        self._mpr_channel_transitions_counter = Counter({
            # key: (nb packets transmitted, nb packets successfully delivered)
            # value: nb of occurences
            (n_packets_transmitted, n_packets_delivered): self.prior_dirichlet_concentration
            for n_packets_transmitted in range(n_agents + 1)
            for n_packets_delivered in range(n_packets_transmitted + 1)
        })

    @property
    def data_generation_posterior_params(self):
        return {
            probability_map_name: {
                encode_integers_tuple(adjacent_values, empty_tuple_encoding=""): {
                    encode_integers_tuple(joint_values): self._data_generation_transitions_counters[
                        probability_map_name
                    ][
                        (*joint_values, *adjacent_values)
                    ]
                    for joint_values in product(*[[0, 1] for _ in range(n_covariates["n_joint_agents"])])
                }
                for adjacent_values in product(*[[0, 1] for _ in range(n_covariates["n_adjacent_agents"])])
            }
            for probability_map_name, n_covariates in
            self._data_generation_map_names_to_dirichlet_covariate_length.items()
        }

    @property
    def mpr_channel_posterior_params(self):
        return {
            encode_integers_tuple(transition_tuple): count
            for transition_tuple, count in self._mpr_channel_transitions_counter.items()
        }

    def _add_value_all_keys_counter(self, c: Counter, val: float) -> Counter:
        val_counter = Counter(dict.fromkeys(c.keys(), val))
        return c + val_counter

    def update_prior_dirichlet_concentration(self, new_prior_dirichlet_concentration: float):
        # Update posterior params
        prior_concentration_shift = new_prior_dirichlet_concentration - self.prior_dirichlet_concentration
        self._mpr_channel_transitions_counter = self._add_value_all_keys_counter(
            self._mpr_channel_transitions_counter,
            prior_concentration_shift
        )
        self._data_generation_transitions_counters = {
            probability_map_name: self._add_value_all_keys_counter(counter, prior_concentration_shift)
            for probability_map_name, counter in self._data_generation_transitions_counters.items()
        }

        # Update prior
        self.prior_dirichlet_concentration = new_prior_dirichlet_concentration

    def _update_data_geneneration_posterior(self, transitions: List[Transition]):
        # Count data generation transitions
        for agents_dependency in self._data_generation_dependencies_info:
            # Count transitions for the specific agents dependency
            _formatted_dependecy_transitions = [
                (
                    *(  # Joint agents state at t+1
                        transition.agents_next_observations[joint_agent_idx].data_input
                        for joint_agent_idx in agents_dependency.joint_agents
                    ),
                    *(  # Adjacent agents at t
                        transition.agents_observations[adjacent_agent_idx].data_input
                        for adjacent_agent_idx in agents_dependency.adjacent_agents
                    ),
                )
                for transition in transitions
            ]
            count_dependecy_transitions = Counter(_formatted_dependecy_transitions)

            # Update the counter of the given probabilities_map
            self._data_generation_transitions_counters[agents_dependency.probabilities_map_name].update(
                count_dependecy_transitions
            )

    def _update_mpr_channel_posterior(self, transitions: List[Transition]):
        # Count transitions
        _formatted_transitions = [
            (
                sum(transition.agents_actions),
                sum(map(lambda obs: obs.ack, transition.agents_next_observations))
            )
            for transition in transitions
        ]
        count_transitions = Counter(_formatted_transitions)

        # Update Dirichlet posterior
        self._mpr_channel_transitions_counter.update(count_transitions)

    def update_model(self, transitions: List[Transition]):
        self._update_data_geneneration_posterior(transitions)
        self._update_mpr_channel_posterior(transitions)

    def _sample_distribution(self, counts: List[int], model_sampling_method: str) -> List[float]:
        if model_sampling_method == SampleMethod.POSTERIOR_SAMPLE:
            for c in range(NAN_SAMPLE_RETRY):  # Small values in <count> might yield NaN samples, retry at most 10 times
                sample = dirichlet.rvs(counts, size=1)[0]
                if not np.isnan(sample).any():
                    return sample
                logger.warning(f"NaN sample from model, retrying... (attempt {c})")
            raise ValueError(f"Sampled too many NaNs from Dirichlet parameters '{counts}' ({NAN_SAMPLE_RETRY} retries)")
        elif model_sampling_method == SampleMethod.MAXIMUM_LIKELIHOOD:
            return max_likelihood_estimate(
                posterior_dirichlet_params=counts,
                prior_dirichlet_concentration=self.prior_dirichlet_concentration
            )
        elif model_sampling_method == SampleMethod.MAXIMUM_A_POSTERIORI:
            return max_a_posteriori_estimate(posterior_dirichlet_params=counts)
        else:
            raise ValueError(f"Unknown sample distribution method '{model_sampling_method}'")

    def _sample_data_generator_transition(self, model_sampling_method: str) -> DataGeneratorTransition:
        # Sample a model for each data generation probability map
        data_generation_probabilities_maps = []
        for probabilities_map_name, counter in self._data_generation_transitions_counters.items():
            # Compute sampled probabilities map for given probabilities_map_name
            sampled_probabilities_map = dict()
            n_covariates = self._data_generation_map_names_to_dirichlet_covariate_length[probabilities_map_name]
            for adjacent_agents_values in product(*[
                [0, 1] for _ in range(n_covariates["n_adjacent_agents"])
            ]):
                # Sample dirichlet
                all_joint_agents_values = list(product(*[[0, 1] for _ in range(n_covariates["n_joint_agents"])]))
                counter_joint_given_adjacent = [
                    counter[(*joint_agents_values, *adjacent_agents_values)]
                    for joint_agents_values in all_joint_agents_values
                ]
                joint_agents_distribution_given_adjacent = self._sample_distribution(
                    counter_joint_given_adjacent,
                    model_sampling_method
                )

                # Store into probabilities map
                adjacent_agents_key = encode_integers_tuple(
                    adjacent_agents_values,
                    empty_tuple_encoding="default"  # Surrogate key when dependency does not depend on adjacent agents
                )
                sampled_probabilities_map[adjacent_agents_key] = {
                    encode_integers_tuple(joint_agents_values): joint_data_gen_prob
                    for joint_agents_values, joint_data_gen_prob in
                    zip(all_joint_agents_values, joint_agents_distribution_given_adjacent)
                }

            # Store sampled probability map
            if n_covariates["n_adjacent_agents"] == 0:  # Default map only
                data_generation_probabilities_maps.append(
                    DataGenerationProbabilitiesMap(
                        probabilities_map_name,
                        n_covariates["n_joint_agents"],
                        probabilities_map=None,
                        default_joint_distribution=sampled_probabilities_map["default"]
                    )
                )
            else:  # Adjacent agents map
                data_generation_probabilities_maps.append(
                    DataGenerationProbabilitiesMap(
                        probabilities_map_name,
                        n_covariates["n_joint_agents"],
                        probabilities_map=sampled_probabilities_map,
                        default_joint_distribution=None
                    )
                )

        # Return generator with given dependencies and sampled probabilities maps
        return DataGeneratorTransition(
            self.n_agents,
            self._data_generation_dependencies_info,
            data_generation_probabilities_maps
        )

    def _sample_mpr_channel_matrix(self, model_sampling_method: str) -> np.ndarray:
        mpr_channel_matrix = np.zeros((self.n_agents + 1, self.n_agents + 1))
        for n_packets_transmitted in range(self.n_agents + 1):
            counter_given_packets_transmitted = [
                self._mpr_channel_transitions_counter[(n_packets_transmitted, n_packets_delivered)]
                for n_packets_delivered in range(n_packets_transmitted + 1)
            ]
            mpr_channel_matrix[
                n_packets_transmitted, :(n_packets_transmitted+1)
            ] = self._sample_distribution(
                counter_given_packets_transmitted,
                model_sampling_method
            )
        return mpr_channel_matrix

    def init_env(self, n_packets: int, reward_settings: RewardSettings, model_sampling_method: str) -> MACEnv:
        data_generator_transition = self._sample_data_generator_transition(model_sampling_method)
        data_generator = DataGenerator(self.n_agents, n_packets, data_generator_transition)
        mpr_channel_matrix = self._sample_mpr_channel_matrix(model_sampling_method)
        mpr_channel = MPRChannel(self.n_agents, mpr_channel_matrix)
        agents = [
            Agent(
                agent_index=i,
                n_packets_max=self.n_packets_max_agents,
                reward_settings=reward_settings
            )
            for i in range(self.n_agents)
        ]
        return MACEnv(self.n_agents, data_generator, mpr_channel, agents)

    def update_env(self, current_env: MACEnv, model_sampling_method: str):
        # Instead of resetting the whole MACEnv, we just update the necessary components inplace
        data_generator_transition = self._sample_data_generator_transition(model_sampling_method)
        mpr_channel_matrix = self._sample_mpr_channel_matrix(model_sampling_method)
        current_env.update_dynamics(data_generator_transition, mpr_channel_matrix)

    def save(self, experiment_name: str):
        folder = os.path.join(DATA_FOLDER, experiment_name, ENVIRONMENT_MODEL_FOLDER)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, "env_model.pickle")
        with open(filepath, "wb") as file:
            pickle.dump(self.__dict__, file)

    @classmethod
    def load(cls, experiment_name: str):
        folder = os.path.join(DATA_FOLDER, experiment_name, ENVIRONMENT_MODEL_FOLDER)
        filepath = os.path.join(folder, "env_model.pickle")
        with open(filepath, "rb") as file:
            class_dict = pickle.load(file)
        class_obj = cls(
            class_dict["n_agents"],
            class_dict["n_packets_max_agents"],
            class_dict["_data_generation_dependencies_info"]
        )
        class_obj.__dict__.update(class_dict)
        return class_obj
