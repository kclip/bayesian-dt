import json
import os
from collections import namedtuple
from dataclasses import dataclass, asdict, field, fields
from typing import List

from settings import EXPERIMENT_PHASE_EPISODE_FOLDER, DATA_FOLDER, Metadata
from src.utils import JSONNumpyEncoder


Observation = namedtuple(
    "Observation",
    [
        "n_packets_max",
        "n_packets_buffer",
        "data_input",
        "ack",
        "time_step"
    ]
)


@dataclass()
class Transition(object):
    agents_observations: list[Observation]
    agents_actions: list[int]
    agents_next_observations: list[Observation]
    agents_rewards: list[float]


AgentTransition = namedtuple("AgentTransition", ["observation", "action", "next_observation", "reward"])


@dataclass()
class RewardSettings(object):
    cooperative_reward: bool
    reward_ack: float
    reward_overflow: float
    buffer_penalty_amplitude: float
    reward_collision: float
    reward_default: float


@dataclass()
class State(object):
    channel_ack: list[int]
    data_generated: list[int]
    agents_buffer: list[int]


@dataclass()
class StepInfo(object):
    buffer_overflow: list[int] = field(default_factory=list)
    channel_collision: list[int] = field(default_factory=list)
    end_episode: int = 0
    other: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data):
        class_fields = {f.name for f in fields(cls) if f.name != "other"}
        class_data = {
            "other": dict()
        }
        for k, v in data.items():
            if k in class_fields:
                class_data[k] = v
            else:
                class_data["other"][k] = v
        return cls(**class_data)


@dataclass()
class Step(object):
    rewards: list[float]
    state: State
    actions: list[int]
    info: StepInfo
    digital_twin_info: field(default_factory=dict)
    train_info: field(default_factory=dict)


class Episode(object):
    def __init__(self, n_episode: int, metadata: Metadata, history: List[Step] = None):
        self.n_episode = n_episode
        self.metadata: Metadata = metadata
        self.history: List[Step] = history or []

    def add_step(self, step: Step):
        self.history.append(step)

    def _to_json(self):
        return {
            "n_episode": self.n_episode,
            "metadata": asdict(self.metadata),
            "history": [asdict(step) for step in self.history]
        }

    @staticmethod
    def _from_json(data: dict, light_selection: dict = None):
        metadata = Metadata.from_dict(data["metadata"])
        if light_selection is None:
            history = [
                Step(
                    rewards=step["rewards"],
                    actions=step["actions"],
                    info=StepInfo.from_dict(step["info"]),
                    state=State(**step["state"]),
                    digital_twin_info=step.get("digital_twin_info", dict()),
                    train_info=step.get("train_info", dict()),
                )
                for step in data["history"]
            ]
        else:
            history = [
                Step(
                    rewards=step["rewards"] if light_selection.get("rewards", False) else None,
                    actions=step["actions"] if light_selection.get("actions", False) else None,
                    info=StepInfo.from_dict({
                        k: v
                        for k, v in step["info"].items()
                        if light_selection.get("info", dict()).get(k, False)
                    }),
                    state=State(**step["state"]) if light_selection.get("state", False) else None,
                    digital_twin_info={
                        k: v
                        for k, v in step.get("digital_twin_info", dict()).items()
                        if light_selection.get("digital_twin_info", dict()).get(k, False)
                    },
                    train_info={
                        k: v
                        for k, v in step.get("train_info", dict()).items()
                        if light_selection.get("train_info", dict()).get(k, False)
                    },
                )
                for step in data["history"]
            ]
        return data["n_episode"], metadata, history

    def save_episode(self, experiment_name: str, episode_name: str, experiment_phase: str = "TEST_POLICY"):
        episode_folder = EXPERIMENT_PHASE_EPISODE_FOLDER[experiment_phase]
        experiment_directory = os.path.join(DATA_FOLDER, experiment_name, episode_folder)
        if not os.path.isdir(experiment_directory):
            os.makedirs(experiment_directory)
        filepath = os.path.join(experiment_directory, f"{episode_name}.json")
        if os.path.isfile(filepath):
            raise FileExistsError(f"Episode log file '{experiment_name}' already exists")
        data = self._to_json()
        with open(filepath, "w") as file:
            json.dump(data, file, cls=JSONNumpyEncoder)

    @classmethod
    def load_episode(
        cls,
        experiment_name: str,
        episode_name: str,
        experiment_phase: str = "TEST_POLICY",
        light_selection: dict = None
    ):
        episode_folder = EXPERIMENT_PHASE_EPISODE_FOLDER[experiment_phase]
        filepath = os.path.join(DATA_FOLDER, experiment_name, episode_folder, f"{episode_name}.json")
        if not os.path.isfile(filepath):
            raise FileNotFoundError("File not found")
        with open(filepath, "r") as file:
            n_episode, metadata, history = cls._from_json(json.load(file), light_selection=light_selection)
            file.close()
        return cls(n_episode, metadata, history=history)

    @classmethod
    def load_experiment(
        cls,
        experiment_name: str,
        experiment_phase: str = "TEST_POLICY",
        light_selection: dict = None,
        select_episodes: List[str] = None
    ):
        episode_folder = EXPERIMENT_PHASE_EPISODE_FOLDER[experiment_phase]
        experiment_dir = os.path.join(DATA_FOLDER, experiment_name, episode_folder)
        episode_names = select_episodes or [
            episode_filename.split(".")[0] for episode_filename in os.listdir(experiment_dir)
        ]
        episodes = [
            cls.load_episode(
                experiment_name,
                episode_name,
                experiment_phase=experiment_phase,
                light_selection=light_selection
            )
            for episode_name in episode_names
        ]
        return sorted(episodes, key=lambda x: x.n_episode)
