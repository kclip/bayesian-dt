# Experimental setup
# ------------------
# This files defines all the constants used to design the experimental setup

# Simulation
N_EPISODES_TEST = 3
N_STEPS_TEST = 8000
N_EPISODES_TRAINING_POLICY = 60
N_STEPS_TRAINING_POLICY = 2000
N_AGENTS = 4
N_PACKETS = 10 * N_STEPS_TEST * N_AGENTS  # i.e. we never run out of packets during training or testing

# Buffer capacity
N_PACKETS_MAX = 1

# Data generation probabilities
DATA_GEN_PROBABILITIES_MAP = None  # No time-dependent data generation
# Values here describe the approximately the setup above
DEFAULT_JOINT_DISTRIBUTION = {
    "0,0": 0.2,
    "1,0": 0.4,
    "0,1": 0.4,
    "1,1": 0
}
DATA_GEN_PROBABILITIES_MAPS_KWARGS = [
    {
        "name": "cluster_1",
        "n_joint_agents": 2,
        "probabilities_map": DATA_GEN_PROBABILITIES_MAP,
        "default_joint_distribution": DEFAULT_JOINT_DISTRIBUTION
    },
    {
        "name": "cluster_2",
        "n_joint_agents": 2,
        "probabilities_map": DATA_GEN_PROBABILITIES_MAP,
        "default_joint_distribution": DEFAULT_JOINT_DISTRIBUTION
    }
]

DATA_GEN_DEPENDENCIES_KWARGS = [
    {
        "joint_agents": [0, 1],
        "adjacent_agents": [],
        "probabilities_map_name": "cluster_1"
    },
    {
        "joint_agents": [2, 3],
        "adjacent_agents": [],
        "probabilities_map_name": "cluster_2"
    },
]


# MPR channel
# Note: this MPR channel has an average transmission of 1.2 packets when two packets are sent
MPR_MATRIX = [
    [1, 0, 0, 0, 0],  # 0 packets sent
    [0, 1, 0, 0, 0],  # 1 packets sent -> no collision
    [0, 0.8, 0.2, 0, 0],  # 2 packets sent -> 2 packets can go through
    [1, 0, 0, 0, 0],  # for 3 packets packets sent or more -> collision
    [1, 0, 0, 0, 0],
]


# Reward conf
COOPERATIVE_REWARD = True
REWARD_ACK = 50  # The buffer overflow penalty is taken as the oppositve of this
REWARD_DEFAULT = -1
RETURN_DISCOUNT = 0.95  # After 100 steps (RETURN_DISCOUNT ** 100), we only consider less than ~0.5% of the reward


# Model based
PRIOR_DIRICHLET_CONCETRATION = 0.01
PRIOR_DIRICHLET_CONCETRATION_MAP = 1.01  # MAP needs to have Dir parameters strictly above 1, otherwise it's ill defined
N_STEPS_BETWEEN_POSTERIOR_SAMPLE = 200


# ENV METADATA
ENV_METADATA = {
    "n_agents": N_AGENTS,
    "n_packets": N_PACKETS,
    "test_n_episodes": N_EPISODES_TEST,
    "test_max_steps": N_STEPS_TEST,
    # Collision channel
    "mpr_matrix": MPR_MATRIX,
    # Data generation
    "data_generator_probabilities_maps_kwargs": DATA_GEN_PROBABILITIES_MAPS_KWARGS,
    "data_generator_dependencies_kwargs": DATA_GEN_DEPENDENCIES_KWARGS,
    # Agents
    "n_packets_max": N_PACKETS_MAX,
    "use_legacy_reward": False,
    # Reward
    "cooperative_reward": COOPERATIVE_REWARD,
    "reward_ack": REWARD_ACK,
    "reward_overflow": -REWARD_ACK,
    "buffer_penalty_amplitude": 0,
    "reward_collision": REWARD_DEFAULT,
    "reward_default": REWARD_DEFAULT,
}


# For optimizers
BATCH_SIZE = 32
LAMBDA_RETURN_N_STEPS = 20
LAMBDA_RETURN_DECAY = 0.8
MIN_BATCH_SIZE = BATCH_SIZE + LAMBDA_RETURN_N_STEPS
MAX_BATCH_SIZE = BATCH_SIZE + LAMBDA_RETURN_N_STEPS + 100
TDMA_FRAME_LENGTH = 4

# COMA Policy metadata base
POLICY_OPTIMIZER_METADATA = {
    "return_discount": RETURN_DISCOUNT,
    "batch_size": BATCH_SIZE,
    # Buffer
    "transitions_buffer_min_size": MIN_BATCH_SIZE,
    "transitions_buffer_max_size": MAX_BATCH_SIZE,
    # TDMA CONF
    "frame_length": N_AGENTS,  # 1 slot per agent
    # Actor
    "actor_optimizer_name": "rmsprop",
    "actor_architecture_name": "PolicySlottedNN",
    "share_actor_network": False,
    "learning_rate_actor": 0.0002,
    "actor_optimizer_kwargs": {},
    "n_critic_updates_between_actor_updates": 1,
    # Critic
    "critic_optimizer_name": "rmsprop",
    "critic_architecture_name": "DecomposedQValueNN",
    "learning_rate_critic": 0.001,
    "critic_optimizer_kwargs": {},
    "n_updates_between_critic_target_updates": 5,
    "lambda_return_decay": LAMBDA_RETURN_DECAY,
    "lambda_return_n_steps": LAMBDA_RETURN_N_STEPS,
    # Entropy exploration
    "entropy_exploration_n_steps": 20 * (N_STEPS_TRAINING_POLICY // (MIN_BATCH_SIZE - 1)),  # 20 episodes
    "entropy_exploration_temperature": 50,
    # Exploration
    "epsilon_exploration_start": 0,
    "epsilon_exploration_end": 0,
    "epsilon_exploration_step": 0.9997,
    "epsilon_exploration_update_frequency": 1,
}

# Plot parameters
ROLLING_MEAN_WINDOW = 500
N_STEPS_RETURN_PLOT = 100
