import os
from typing import List

from settings import PROJECT_FOLDER, DATA_FOLDER, POLICY_OPTIMIZER_FOLDER
from src.data_classes import Transition
from src.policy.policy import Policy


class PolicyOptimizer(object):
    def __init__(self):
        pass

    def get_agents_policies(self, training_policy: bool = False) -> List[Policy]:
        raise NotImplementedError()

    def train_step(self, step: int, transitions: List[Transition]) -> dict:
        return dict()

    def reset(self):
        pass

    @staticmethod
    def policy_optimizer_folder(experiment_name: str, create_if_not_exists: bool = True):
        folder = os.path.join(PROJECT_FOLDER, DATA_FOLDER, experiment_name, POLICY_OPTIMIZER_FOLDER)
        if (not os.path.isdir(folder)) and create_if_not_exists:
            os.makedirs(folder)
        return folder

    def save(self, experiment_name: str):
        pass

    @classmethod
    def load(cls, experiment_name: str):
        pass
