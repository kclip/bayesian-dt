import logging
import datetime
from dataclasses import asdict
from settings import ExperimentPhase, Metadata, EnvMetadata
from bin.utils import check_metadata_consistency, check_step_consistency, get_max_abs_reward
from src.data_classes import Episode, Step, State, StepInfo, Transition, RewardSettings
from src.environment.data_generator.data_generator import DataGenerator
from src.environment.data_generator.transition import DataGeneratorDependencyInfo, DataGenerationProbabilitiesMap, \
    DataGeneratorTransition
from src.environment.channel.channel import MPRChannel
from src.environment.agent.agent import Agent
from src.environment.environment import MACEnv
from src.policy.aloha.aloha_optimizer import PolicyOptimizerAloha
from src.policy.tdma.tdma_optimizer import PolicyOptimizerTDMA
from src.policy.aac.aac_optimizer import PolicyOptimizerCommonAAC
from src.policy.coma.coma_optimizer import PolicyOptimizerCOMA
from src.policy.dqn.dqn_optimizer import PolicyOptimizerCommonDQN
from src.digital_twin.digital_twin import DigitalTwin, DigitalTwinPolicyPassthrough, DigitalTwinConcurrentModel, \
    DigitalTwinSeparateModel
from src.view.metrics import throughput_of_agents, get_n_packets_dropped, cumulated_average_reward, get_n_collisions, \
    get_avg_packet_generation


logger = logging.getLogger(__name__)


def run_simulation(
        metadata: Metadata,
        log: bool = False,
        log_train: bool = False,
        log_experiment_name: str = None,
        load_forced_test_experiment_name: str = None,
        load_model_experiment_name: str = None,
        load_policy_experiment_name: str = None,
        suffix_log_experiment_name: bool = False,
        log_trained_model_or_policy: bool = True,
        log_model_at_each_step: bool = False
):
    # Check config
    check_metadata_consistency(metadata)

    # INIT SIMULATION
    # ---------------

    if suffix_log_experiment_name:
        run_timestamp_suffix = datetime.datetime.now().strftime("%Y%m%d%H%M")
        log_experiment_name = f"{log_experiment_name}_{run_timestamp_suffix}"
        logger.info(f"Experiment log name : {log_experiment_name}")

    # Force current experiment to have the same data generation states over time as the specified experiment for
    # all episodes
    # Note: environment metadata is set to match the first scenario, thus episodes in a single experiment should all
    # have the same environment metadata
    forced_test_experiment = None
    if load_forced_test_experiment_name:
        forced_test_experiment = Episode.load_experiment(load_forced_test_experiment_name)
        metadata.env_metadata = EnvMetadata(
            **asdict(forced_test_experiment[0].metadata.env_metadata)
        )

    # Environment size
    n_agents = metadata.env_metadata.n_agents
    n_packets = metadata.env_metadata.n_packets

    # Data generator
    data_gen_probabilities_maps = [
        DataGenerationProbabilitiesMap(
            map_kwargs["name"],
            map_kwargs["n_joint_agents"],
            probabilities_map=map_kwargs["probabilities_map"],
            default_joint_distribution=map_kwargs["default_joint_distribution"]
        )
        for map_kwargs in metadata.env_metadata.data_generator_probabilities_maps_kwargs
    ]
    data_gen_dependencies = [
        DataGeneratorDependencyInfo(
            joint_agents=dependency_kwarg["joint_agents"],
            adjacent_agents=dependency_kwarg["adjacent_agents"],
            probabilities_map_name=dependency_kwarg["probabilities_map_name"]
        )
        for dependency_kwarg in metadata.env_metadata.data_generator_dependencies_kwargs
    ]
    data_gen_transition = DataGeneratorTransition(
        metadata.env_metadata.n_agents,
        data_gen_dependencies,
        data_gen_probabilities_maps
    )
    data_gen = DataGenerator(n_agents, n_packets, data_gen_transition)

    # MPR Channel
    channel = MPRChannel(
        n_agents,
        metadata.env_metadata.mpr_matrix,
    )

    # Agents
    reward_settings = RewardSettings(
        cooperative_reward=metadata.env_metadata.cooperative_reward,
        reward_ack=metadata.env_metadata.reward_ack,
        reward_overflow=metadata.env_metadata.reward_overflow,
        buffer_penalty_amplitude=metadata.env_metadata.buffer_penalty_amplitude,
        reward_collision=metadata.env_metadata.reward_collision,
        reward_default=metadata.env_metadata.reward_default
    )
    agents = [
        Agent(
            agent_index=i,
            n_packets_max=metadata.env_metadata.n_packets_max,
            reward_settings=reward_settings,
            use_legacy_reward=metadata.env_metadata.use_legacy_reward
        )
        for i in range(n_agents)
    ]

    # Env
    env = MACEnv(n_agents, data_gen, channel, agents)

    # Policies
    policy_optimizer_args = []
    policy_optimizer_class = None
    if metadata.train_metadata.policy_optimizer_class == "PolicyOptimizerAloha":
        policy_optimizer_args = [n_agents]
        policy_optimizer_class = PolicyOptimizerAloha
    elif metadata.train_metadata.policy_optimizer_class == "PolicyOptimizerTDMA":
        policy_optimizer_class = PolicyOptimizerTDMA
    elif metadata.train_metadata.policy_optimizer_class == "PolicyOptimizerCommonAAC":
        policy_optimizer_args = [n_agents, metadata.env_metadata.n_packets_max]
        policy_optimizer_class = PolicyOptimizerCommonAAC
    elif metadata.train_metadata.policy_optimizer_class == "PolicyOptimizerCOMA":
        max_abs_reward = get_max_abs_reward(
            reward_settings,
            [metadata.env_metadata.n_agents],
            metadata.env_metadata.n_packets_max
        )
        policy_optimizer_args = [n_agents, metadata.env_metadata.n_packets_max, max_abs_reward]
        policy_optimizer_class = PolicyOptimizerCOMA
    elif metadata.train_metadata.policy_optimizer_class == "PolicyOptimizerCommonDQN":
        policy_optimizer_args = [n_agents, metadata.env_metadata.n_packets_max]
        policy_optimizer_class = PolicyOptimizerCommonDQN
    else:
        raise ValueError(f"Unknown policy class '{metadata.train_metadata.policy_optimizer_class}'")

    if load_policy_experiment_name is None:  # Init optimizer / policies
        logger.info("====== INIT POLICY ======")
        policy_optimizer = policy_optimizer_class(
            *policy_optimizer_args,
            **metadata.train_metadata.policy_optimizer_kwargs
        )
    else:  # Load optimizer / policies
        logger.info("====== LOADING POLICY ======")
        policy_optimizer = policy_optimizer_class.load(load_policy_experiment_name)

    # Digital Twin
    digital_twin_class = None
    digital_twin_args = []
    if metadata.train_metadata.digital_twin_class == "DigitalTwinPolicyPassthrough":
        digital_twin_class = DigitalTwinPolicyPassthrough
    elif metadata.train_metadata.digital_twin_class == "DigitalTwinConcurrentModel":
        digital_twin_class = DigitalTwinConcurrentModel
        digital_twin_args = [
            metadata.env_metadata.n_agents,
            metadata.env_metadata.n_packets_max,
            reward_settings,
            data_gen_transition.get_dependencies_info()
        ]
    elif metadata.train_metadata.digital_twin_class == "DigitalTwinSeparateModel":
        digital_twin_class = DigitalTwinSeparateModel
        digital_twin_args = [
            metadata.env_metadata.n_agents,
            metadata.env_metadata.n_packets_max,
            reward_settings,
            data_gen_transition.get_dependencies_info()
        ]
    else:
        raise ValueError(f"Unknown digital twin class '{metadata.train_metadata.digital_twin_class}'")
    digital_twin = digital_twin_class(
        policy_optimizer,
        *digital_twin_args,
        **metadata.train_metadata.digital_twin_kwargs
    )
    if load_model_experiment_name is not None:
        logger.info("====== LOAD ENVIRONMENT MODEL ======")
        digital_twin.load_model(
            load_model_experiment_name,
            prior_dirichlet_concentration=metadata.train_metadata.digital_twin_kwargs.get(
                "prior_dirichlet_concentration", None
            )
        )

    # TRAIN ENVIRONMENT MODEL
    # -----------------------
    if (
        (load_model_experiment_name is None) and
        (metadata.train_metadata.train_model_n_episodes > 0)
    ):
        logger.info("====== TRAINING ENVIRONMENT MODEL ======")
        execute_episodes(
            metadata=metadata,
            env=env,
            digital_twin=digital_twin,
            episodes=metadata.train_metadata.train_model_n_episodes,
            max_steps=metadata.train_metadata.train_model_max_steps,
            experiment_phase=ExperimentPhase.TRAIN_MODEL,
            forced_test_experiment=None,
            log=log_train,
            log_experiment_name=log_experiment_name,
            log_model_at_each_step=log_model_at_each_step
        )
        # Log trained model
        if log_trained_model_or_policy:
            if log_model_at_each_step:
                digital_twin.save_model(
                    f"{log_experiment_name}_n_steps_{metadata.train_metadata.train_model_max_steps}"
                )
            else:
                digital_twin.save_model(log_experiment_name)

    # TRAIN POLICIES
    # --------------
    if (
        (load_policy_experiment_name is None) and
        (metadata.train_metadata.train_policy_n_episodes > 0)
    ):
        logger.info("====== TRAINING POLICIES ======")
        virtual_env = digital_twin.init_virtual_env(env)
        execute_episodes(
            metadata=metadata,
            env=virtual_env,
            digital_twin=digital_twin,
            episodes=metadata.train_metadata.train_policy_n_episodes,
            max_steps=metadata.train_metadata.train_policy_max_steps,
            experiment_phase=ExperimentPhase.TRAIN_POLICY,
            forced_test_experiment=None,
            log=log_train,
            log_experiment_name=log_experiment_name,
        )
        # Log trained policy
        if log_trained_model_or_policy:
            digital_twin.policy_optimizer.save(log_experiment_name)

    # TEST POLICIES
    # -----------
    if metadata.env_metadata.test_n_episodes > 0:
        logger.info("====== TESTING POLICIES ======")
        # Prepare policy for testing (remove potential exploration, ...)
        digital_twin.post_training()
        # Test episodes
        execute_episodes(
            metadata=metadata,
            env=env,
            digital_twin=digital_twin,
            episodes=metadata.env_metadata.test_n_episodes,
            max_steps=metadata.env_metadata.test_max_steps,
            experiment_phase=ExperimentPhase.TEST_POLICY,
            forced_test_experiment=forced_test_experiment,
            log=log,
            log_experiment_name=log_experiment_name,
        )

    return log_experiment_name


def execute_episodes(
    metadata: Metadata,
    env: MACEnv,
    digital_twin: DigitalTwin,
    episodes: int,
    max_steps: int,
    experiment_phase: str,
    forced_test_experiment=None,
    log: bool = False,
    log_experiment_name: str = None,
    log_model_at_each_step: bool = False
):
    log_freq = max(1, max_steps // 10)
    for n_episode in range(episodes):
        # Reset environment and internal states
        agents_observations = env.reset()
        digital_twin.reset()
        agents_policies = digital_twin.get_agents_policies()  # Initiate the physical twin
        agents_rewards = [0] * metadata.env_metadata.n_agents
        info = {
            "end_episode": 0
        }
        digital_twin_info = digital_twin.get_info()
        train_info = dict()
        step = 0

        # Force data generation and MPR channel states to behave like in the loaded episode
        forced_episode = None
        if forced_test_experiment is not None:
            forced_episode = forced_test_experiment[n_episode]

        # Init data logger
        episode_data = Episode(n_episode, metadata)

        # Play simulation
        # Note: we do not stop when episode terminates because we might still need to log the whole scenario
        while step < max_steps:
            env.render()

            if step % log_freq == 0:
                logger.debug(f"----- Step t = {str(step)} -----")
                logger.debug(f"N packets left : {env.data_gen.n_packets_left}")
                logger.debug(f"Data in t : {[obs.data_input for obs in agents_observations]}")
                logger.debug(f"Buffers t : {[obs.n_packets_buffer for obs in agents_observations]}")

            # Agents actions at time step t
            agents_actions = [
                agent.action(observation)[0]
                for agent, observation in zip(agents_policies, agents_observations)
            ]

            check_step_consistency(agents_observations, agents_actions)

            # Store time step t data
            step_data = Step(
                rewards=agents_rewards,
                state=State(
                    channel_ack=env.channel.ack_state,
                    data_generated=env.data_gen.data_input_state,
                    agents_buffer=[agent.n_packets_buffer for agent in env.agents]
                ),
                actions=agents_actions,
                info=StepInfo.from_dict(info),
                digital_twin_info=digital_twin_info,
                train_info=train_info
            )
            episode_data.add_step(step_data)

            # Transition to time step t+1 : state_t+1 and reward_t+1
            forced_episode_next_step = (
                forced_episode.history[step+1] if forced_episode is not None else None
            )
            agents_next_observations, agents_rewards, done, info = env.step(
                agents_actions,
                forced_episode_next_step=forced_episode_next_step,
                cooperative_reward=metadata.env_metadata.cooperative_reward
            )
            # Note: technically the reward should be defined at the digital twin

            # Centralized Training with Decentralized Execution (CTDE):
            # During training, the digital twin has access to the state (i.e. all the observations here). It trains
            # using the received data and issues a new set of policies to the physical twin
            if (
                (experiment_phase == ExperimentPhase.TRAIN_MODEL) or
                (experiment_phase == ExperimentPhase.TRAIN_POLICY)
            ):
                transition = Transition(
                    agents_observations=agents_observations,
                    agents_actions=agents_actions,
                    agents_next_observations=agents_next_observations,
                    agents_rewards=agents_rewards,
                )
                if experiment_phase == ExperimentPhase.TRAIN_MODEL:
                    if log_model_at_each_step:
                        digital_twin.save_model(f"{log_experiment_name}_n_steps_{step}")
                    # Train model and update exploration policy
                    train_info = digital_twin.train_model_step(step, [transition])
                    agents_policies = digital_twin.get_agents_policies()
                if experiment_phase == ExperimentPhase.TRAIN_POLICY:
                    # Train and update agents
                    train_info = digital_twin.train_policy_step(step, [transition])
                    agents_policies = digital_twin.get_agents_policies()
                    # Update env (e.g. : sample new env dynamics, do nothing, ...)
                    digital_twin.update_virtual_env(step, env)
                digital_twin_info = digital_twin.get_info()

            # Update info
            info["end_episode"] = int(done)

            if step % log_freq == 0:
                logger.debug(f"Actions t : {list(agents_actions)}")
                logger.debug(f"ACK in t+1: {[int(obs.ack) for obs in agents_observations]}")
                logger.debug(f"Reward per agent {sum(agents_rewards) / len(agents_rewards)}")

            # Update temp vars
            step += 1
            agents_observations = agents_next_observations

        # Store last step
        step_data = Step(
            rewards=agents_rewards,
            state=State(
                channel_ack=env.channel.ack_state,
                data_generated=env.data_gen.data_input_state,
                agents_buffer=[agent.n_packets_buffer for agent in env.agents]
            ),
            actions=[0] * env.n_agents,
            info=StepInfo.from_dict(info),
            digital_twin_info=digital_twin_info,
            train_info=train_info
        )
        episode_data.add_step(step_data)

        # Store episode
        if log:
            episode_name = f"ep_{n_episode}"
            episode_data.save_episode(log_experiment_name, episode_name, experiment_phase=experiment_phase)
            logger.info(f"Episode successfully logged in {log_experiment_name}/{episode_name}")

        # Log episode metrics
        max_throughput, agents_throughput, n_packets_sent = throughput_of_agents(episode_data)
        n_packets_dropped = get_n_packets_dropped(episode_data)
        score = cumulated_average_reward(episode_data)
        n_collisions = get_n_collisions(episode_data)
        avg_packet_generation = get_avg_packet_generation(episode_data)
        logger.info(
            f"Episode {n_episode} : mean cumulated reward per agent {score}, "
            f"{avg_packet_generation:.4f} mean packets generated per step per agent, "
            f"{n_packets_sent} packets sent, "
            f"{n_packets_dropped} packets dropped, "
            f"{n_collisions} collisions"
        )

    env.close()
