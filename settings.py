import os
from dataclasses import dataclass, field
import torch


# Matplotlib config
# -----------------
def setup_matplotlib_config():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.set_loglevel("INFO")
    sns.set_theme()  # Set matplotlib plots to have seaborn's plot theme
    mpl.rcParams['font.family'] = "Times New Roman"


# Pytorch
# -------
DEVICE = torch.device("cpu")  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Data logging
# ------------
PROJECT_FOLDER = os.path.dirname(os.path.realpath(__file__))
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'logs')
TEST_EPISODES_FOLDER = "test_episodes"
TRAIN_MODEL_TEST_EPISODES_FOLDER = "train_model_episodes"
TRAIN_POLICY_TEST_EPISODES_FOLDER = "train_policy_episodes"
POLICY_OPTIMIZER_FOLDER = "policy_optimizer"
ENVIRONMENT_MODEL_FOLDER = "environment_model"


# Experiment phases
# -----------------
@dataclass(frozen=True)
class ExperimentPhase(object):
    TRAIN_MODEL = "TRAIN_MODEL"
    TRAIN_POLICY = "TRAIN_POLICY"
    TEST_POLICY = "TEST_POLICY"


EXPERIMENT_PHASE_EPISODE_FOLDER = {
    ExperimentPhase.TRAIN_MODEL: TRAIN_MODEL_TEST_EPISODES_FOLDER,
    ExperimentPhase.TRAIN_POLICY: TRAIN_POLICY_TEST_EPISODES_FOLDER,
    ExperimentPhase.TEST_POLICY: TEST_EPISODES_FOLDER,
}


# Simulation parameters
# ---------------------

@dataclass()
class EnvMetadata(object):
    # General
    n_agents: int = 10
    n_packets: int = 10000

    # Simulation validation
    test_n_episodes: int = 10
    test_max_steps: int = 1000

    # Data generator
    data_generator_probabilities_maps_kwargs: list = field(default_factory=lambda: list())
    data_generator_dependencies_kwargs: list = field(default_factory=lambda: list())

    # MPR Channel
    mpr_matrix: list = field(default_factory=lambda: list())  # Matrix of size (n_agents + 1) x (n_agents + 1)

    # Agent
    n_packets_max: int = 3  # Buffer size agent
    use_legacy_reward: bool = True  # Use old or new reward function.
    # Reward
    cooperative_reward: bool = False
    reward_ack: int = 0  # Reward for transmission success
    reward_overflow: int = 0  # Penalty for buffer overflow
    buffer_penalty_amplitude: int = 0  # Amplitude for buffer filling penalty, scales linearly with the number of
    #                                    packets in buffer from 0 to buffer_penalty_amplitude
    reward_collision: int = 0  # Penalty for sending a packet resulting in a channel collision
    reward_default: int = 0  # Reward if none of the above happens


# Default policy parameters
# -------------------------

@dataclass()
class TrainMetadata(object):
    # Digital Twin
    digital_twin_class: str = "DigitalTwinPolicyPassthrough"
    digital_twin_kwargs: dict = field(default_factory=dict)

    # Policies
    policy_optimizer_class: str = "PolicyOptimizerCommonAAC"
    policy_optimizer_kwargs: dict = field(default_factory=dict)

    # Model training
    train_model_n_episodes: int = 10
    train_model_max_steps: int = 1000
    
    # Policy Training
    train_policy_n_episodes: int = 10
    train_policy_max_steps: int = 1000


# All metadata
# ------------

@dataclass()
class Metadata(object):
    env_metadata: EnvMetadata = field(default_factory=EnvMetadata)
    train_metadata: TrainMetadata = field(default_factory=TrainMetadata)

    @classmethod
    def from_dict(cls, data):
        return cls(
            env_metadata=EnvMetadata(**data.get("env_metadata", dict())),
            train_metadata=TrainMetadata(**data.get("train_metadata", dict())),
        )
