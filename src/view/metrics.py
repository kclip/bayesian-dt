from typing import List, Tuple
import numpy as np
import pandas as pd
import scipy.stats as st
from itertools import product

from src.data_classes import Episode, Transition, Observation


def throughput_of_agents(episode: Episode):
    max_throughput = []
    agents_throughput = []
    for step in episode.history:
        max_throughput.append(
            sum(step.state.data_generated)
        )
        agents_throughput.append(
            sum(step.state.channel_ack)
        )
    # Align ACK and data generation
    agents_throughput = agents_throughput[1:] + [0]

    n_packets_sent = sum(agents_throughput)

    return max_throughput, agents_throughput, n_packets_sent


def get_n_packets_dropped(episode: Episode) -> int:
    buffer_overflow = [
        sum(step.info.buffer_overflow) for step in episode.history
    ]
    return sum(buffer_overflow)


def get_n_collisions(episode: Episode) -> int:
    collisions = [
        sum(step.info.channel_collision) for step in episode.history
    ]
    return sum(collisions)


def get_n_packets_sent(episode: Episode) -> int:
    packets_sent = [
        sum(step.state.channel_ack) for step in episode.history
    ]
    return sum(packets_sent)


def get_avg_packet_generation(episode: Episode) -> float:
    packets_generated = np.array([
        step.state.data_generated for step in episode.history
    ])
    return np.mean(packets_generated)


def get_joint_policy_action(episode: Episode):
    return [
        sum(step.actions) for step in episode.history
    ]


def cumulated_average_reward(episode: Episode) -> float:
    score = 0
    for step in episode.history:
        score += sum(step.rewards) / len(step.rewards)
    return round(score, 2)


def experiment_score_metrics(experiment: List[Episode]):
    scores = []
    for episode in experiment:
        scores.append(cumulated_average_reward(episode))
    return np.average(scores), np.std(scores)


def experiment_throughput_mean_dev(experiment_episodes: List[Episode], confidence_level=0.95) -> Tuple[float, float]:
    throughput = []
    for episode in experiment_episodes:
        throughput += [
            sum(step.state.channel_ack) for step in episode.history
        ]
    mean = np.mean(throughput)
    ci = st.norm.interval(alpha=confidence_level, loc=mean, scale=st.sem(throughput))
    dev = (ci[1] - ci[0]) / 2
    return mean, dev


def get_experiment_throughput(experiment_episodes: List[Episode]) -> pd.DataFrame:
    list_df = []

    for n_episode, episode in enumerate(experiment_episodes):
        # Get per step throughput
        df_ep = pd.DataFrame([
            {
                "step": n_step,
                "episode": n_episode,
                "throughput": sum(step.state.channel_ack)
            }
            for n_step, step in enumerate(episode.history)
        ])

        # Append
        list_df.append(df_ep)

    return pd.concat(list_df).reset_index(drop=True)


def _safe_division(num, den, fillna=None):
    if den != 0:
        return num / den
    return fillna


def jain_fairness(episode: Episode, throughput_rolling_mean_window=None) -> List[float]:
    # Get throughput per agent
    agents_throughput = [
        step.state.channel_ack
        for step in episode.history
    ]

    # Smooth throughput per agent
    df = pd.DataFrame(agents_throughput)
    df = df.rolling(
        throughput_rolling_mean_window, min_periods=0, axis=0
    ).mean().fillna(
        0.0, downcast="float64"
    )

    # Compute Jain's index
    episode_fairness = []
    for idx, agents_smoothed_throughput in df.iterrows():
        squared_throughput = [throughput ** 2 for throughput in agents_smoothed_throughput]
        n_agents = len(agents_smoothed_throughput)
        fairness = _safe_division(
            (sum(agents_smoothed_throughput) ** 2),
            (n_agents * sum(squared_throughput)),
            fillna=0
        )
        episode_fairness.append(fairness)
    return episode_fairness


def get_experiment_fairness(
        experiment_episodes: List[Episode],
        throughput_rolling_mean_window: int = None
) -> pd.DataFrame:
    list_df = []

    for n_episode, episode in enumerate(experiment_episodes):
        # Get per step fairness
        episode_fairness = jain_fairness(episode, throughput_rolling_mean_window=throughput_rolling_mean_window)
        df_ep = pd.DataFrame([
            {
                "step": n_step,
                "episode": n_episode,
                "fairness": fairness
            }
            for n_step, fairness in enumerate(episode_fairness)
        ])

        # Append
        list_df.append(df_ep)

    return pd.concat(list_df).reset_index(drop=True)


def get_return(
        experiment_episodes: List[Episode],
        n_steps_return: int,
        return_discount: float
) -> pd.DataFrame:
    data = []

    # Get instantaneous rewards per time step
    for n_episode, episode in enumerate(experiment_episodes):
        data += [
            (
                n_episode,
                n_step,
                np.mean(step.rewards)
            )
            for n_step, step in enumerate(episode.history)
        ]
    df = pd.DataFrame(data, columns=["episode", "step", "reward"])

    # Compute n-steps return
    df["return"] = df["reward"]
    for k in range(1, n_steps_return):
        df["return"] += (
            df["reward"].shift(-k) * (return_discount ** k)
        ).fillna(
            0, downcast="float64"
        )

    return df


def get_buffer_info(experiment_episodes: List[Episode]) -> pd.DataFrame:
    df_list = []
    for n_episode, episode in enumerate(experiment_episodes):
        df = pd.DataFrame(
            [
                (
                    n_step,
                    n_episode,
                    sum(step.state.agents_buffer) / len(step.state.agents_buffer),
                    max(step.state.agents_buffer),
                    sum(step.info.buffer_overflow)
                )
                for n_step, step in enumerate(episode.history)
            ],
            columns=["step", "episode", "avg_buffer", "max_buffer", "sum_overflow"]
        )
        df["cumsum_overflow"] = df["sum_overflow"].cumsum()
        df["norm_overflow"] = df["sum_overflow"] / len(episode.history[0].state.agents_buffer)
        df_list.append(df)
    return pd.concat(df_list).reset_index(drop=True)


def get_channel_collisions(experiment_episodes: List[Episode]) -> pd.DataFrame:
    df_list = []
    for n_episode, episode in enumerate(experiment_episodes):
        df = pd.DataFrame(
            [
                (
                    n_step,
                    n_episode,
                    max(step.info.channel_collision) if step.info.channel_collision else 0,
                    sum(step.info.channel_collision),
                )
                for n_step, step in enumerate(episode.history)
            ],
            columns=["step", "episode", "has_collision", "sum_collision"]
        )
        df["cumsum_collision"] = df["sum_collision"].cumsum()
        df["norm_collision"] = df["sum_collision"] / len(episode.history[0].state.agents_buffer)
        df_list.append(df)
    return pd.concat(df_list).reset_index(drop=True)


def get_training_info(experiment_episodes: List[Episode]) -> pd.DataFrame:
    data = []
    for n_episode, episode in enumerate(experiment_episodes):
        data += [
            (
                n_episode,
                cumulated_average_reward(episode),
                get_n_packets_sent(episode),
                get_n_packets_dropped(episode),
                get_n_collisions(episode)
            )
        ]
    return pd.DataFrame(
        data,
        columns=["episode", "total_reward_avg_agent", "n_packets_sent", "n_packets_dropped", "n_collisions"]
    )


def get_state_distribution(experiment_episodes: List[Episode]):
    data = []
    for episode in experiment_episodes:
        for step in episode.history:
            for n_agent in range(len(step.state.agents_buffer)):
                data += [
                    (
                        step.state.agents_buffer[n_agent],
                        step.state.data_generated[n_agent],
                        step.state.channel_ack[n_agent],
                    )
                ]
    columns = ["buffer", "data_input", "ack"]
    df = pd.DataFrame(data, columns=columns)
    distributions = dict()
    for col in columns:
        df_tmp = df.groupby(col)[col].count()
        df_tmp = df_tmp / df_tmp.sum()
        distributions[col] = df_tmp
    return distributions


def get_aac_info(experiment_name: str):
    from src.policy.aac.aac_optimizer import PolicyOptimizerCommonAAC

    aac_optimizer = PolicyOptimizerCommonAAC.load(experiment_name)
    aac = aac_optimizer.get_agents_policies()[0]
    data = []
    range_buffer = range(0, aac.n_packets_max + 1)
    for ack, data_in, n_packets_buffer in product([0, 1], [0, 1], range_buffer):
        obs = Observation(
            n_packets_max=aac.n_packets_max,
            ack=ack,
            data_input=data_in,
            n_packets_buffer=n_packets_buffer,
            time_step=0  # Not used in AAC
        )
        actor_prob = aac.get_p_transmit(obs)
        value_critic = aac.get_value_critic(obs)
        data.append((ack, data_in, n_packets_buffer, actor_prob, value_critic))

    return pd.DataFrame(
        data,
        columns=["ack", "data_input", "n_packets_buffer", "actor_p_transmit", "critic_value"]
    )


def get_coma_info(experiment_name: str, samples_critic=100, time_step: int = 0):
    from src.policy.coma.coma_optimizer import PolicyOptimizerCOMA

    coma_optimizer = PolicyOptimizerCOMA.load(experiment_name)
    coma_policies = coma_optimizer.get_agents_policies()
    n_packets_max = coma_policies[0].n_packets_max
    data = []
    range_buffer = range(0, n_packets_max + 1)
    n_agents = coma_optimizer.n_agents
    for ack, data_gen, n_packets_buffer in product([0, 1], [0, 1], range_buffer):
        obs = Observation(
            n_packets_max=n_packets_max,
            ack=ack,
            data_input=data_gen,
            n_packets_buffer=n_packets_buffer,
            time_step=time_step
        )

        actors_prob = [policy.get_p_transmit(obs) for policy in coma_policies]

        agents_observations = [obs] * n_agents
        value_critic = 0
        for _ in range(samples_critic):
            agents_actions = [
                np.random.choice([0, 1], p=[1 - p_transmit, p_transmit])
                for p_transmit in actors_prob
            ]
            value_critic += coma_optimizer.get_value_critic(
                Transition(
                    agents_observations=agents_observations,
                    agents_actions=agents_actions,
                    agents_rewards=[],
                    agents_next_observations=[]
                )
            )
        value_critic = value_critic / samples_critic

        mean_actor_prob = np.mean(actors_prob)

        data.append((ack, data_gen, n_packets_buffer, mean_actor_prob, value_critic))

    return pd.DataFrame(
        data,
        columns=["ack", "data_input", "n_packets_buffer", "actor_p_transmit", "critic_value"]
    )


def get_coma_actors_info(experiment_name: str, time_step: int = 0, is_tdma_actor: bool = False) -> List[pd.DataFrame]:
    from src.policy.coma.coma_optimizer import PolicyOptimizerCOMA

    coma_optimizer = PolicyOptimizerCOMA.load(experiment_name)
    coma_policies = coma_optimizer.get_agents_policies()
    n_packets_max = coma_policies[0].n_packets_max
    all_data = [[] for _ in range(coma_optimizer.n_agents)]
    range_buffer = range(0, n_packets_max + 1)

    for ack, data_gen, n_packets_buffer in product([0, 1], [0, 1], range_buffer):
        obs = Observation(
            n_packets_max=n_packets_max,
            ack=ack,
            data_input=data_gen,
            n_packets_buffer=n_packets_buffer,
            time_step=time_step
        )
        for actor_idx, policy in enumerate(coma_policies):
            prob = policy.get_p_transmit(obs)
            entry = (ack, data_gen, n_packets_buffer, prob)
            if is_tdma_actor:
                p_slot, slot_agnostic_p_transmit = policy.get_p_slot_and_slot_agnostic_p_transmission(obs)
                entry = (*entry, p_slot, slot_agnostic_p_transmit)
            all_data[actor_idx].append(entry)

    columns = ["ack", "data_input", "n_packets_buffer", "actor_p_transmit"]
    if is_tdma_actor:
        columns += ["p_slot", "slot_agnostic_p_transmit"]
    return [
        pd.DataFrame(data, columns=columns)
        for data in all_data
    ]


def get_loss(experiment_train_episodes: List[Episode], loss_name: str) -> pd.DataFrame:
    data = {
        "step": [],
        "loss": []
    }
    for n_episode, episode in enumerate(experiment_train_episodes):
        steps_offset = len(episode.history) * n_episode
        for n_step, step in enumerate(episode.history):
            loss = step.train_info.get(loss_name, None)
            if loss is not None:
                data["step"].append(steps_offset + n_step)
                data["loss"].append(loss)
    return pd.DataFrame(data)


def get_critic_training_info(
        experiment_train_episodes: List[Episode],
        n_steps_return: int,
        return_discount: float
):
    data = {
        "step": [],
        "loss": [],
        "estimated_target": [],
        "estimated_value": [],
        "immediate_reward": [],
        "discounted_n_steps_return": []
    }
    for n_episode, episode in enumerate(experiment_train_episodes):
        steps_offset = len(episode.history) * n_episode
        for n_step, step in enumerate(episode.history):
            loss = step.train_info.get("loss_critic", None)
            if loss is not None:
                data["step"].append(steps_offset + n_step)
                data["loss"].append(loss)
                data["estimated_target"].append(step.train_info["estimated_target_critic"])
                data["estimated_value"].append(step.train_info["estimated_q_value_critic"])
                data["immediate_reward"].append(np.mean(step.rewards))
                discounted_return = 0
                discount_pow_k = 1
                for future_step in episode.history[n_step:n_step + n_steps_return]:
                    discounted_return += discount_pow_k * np.mean(future_step.rewards)
                    discount_pow_k *= return_discount
                data["discounted_n_steps_return"].append(discounted_return)
    return pd.DataFrame(data)


def get_gradients_info(
        experiment_train_episodes: List[Episode],
        gradient_info_name: str,
        gradient_info_num: int = None
):
    column_names = None
    value_names = ["mean_gradient", "max_gradient", "min_gradient"]
    data = []
    for n_episode, episode in enumerate(experiment_train_episodes):
        steps_offset = len(episode.history) * n_episode
        for n_step, step in enumerate(episode.history):
            gradients_info = step.train_info.get(gradient_info_name, None)
            if gradients_info is not None:
                if gradient_info_num is not None:
                    gradients_info = gradients_info[gradient_info_num]
                if column_names is None:
                    column_names = ["step"] + [
                        f"{value_name}_{layer_name}"
                        for value_name, layer_name in product(value_names, gradients_info["layer_name"])
                    ]
                entry = [steps_offset + n_step]
                for value_name in value_names:
                    entry += gradients_info[value_name]
                data.append(entry)
    return pd.DataFrame(data, columns=column_names)


def estimate_Q_value_COMA(
        experiment_name: str,
        n_samples: int = 10,
        min_p_transmit: float = 0,
        max_p_transmit: float = 1,
        n_p_transmit_points: int = 100,
        time_step: int = 0
) -> pd.DataFrame:
    from src.policy.coma.coma_optimizer import PolicyOptimizerCOMA

    coma_optimizer = PolicyOptimizerCOMA.load(experiment_name)
    n_packets_max = coma_optimizer.n_packets_max
    data = []
    range_buffer = range(0, n_packets_max + 1)
    n_agents = coma_optimizer.n_agents
    p_transmit_range = np.linspace(min_p_transmit, max_p_transmit, n_p_transmit_points)
    for ack, data_gen, n_packets_buffer, p_transmit in product([0, 1], [0, 1], range_buffer, p_transmit_range):
        obs = Observation(
            n_packets_max=n_packets_max,
            ack=ack,
            data_input=data_gen,
            n_packets_buffer=n_packets_buffer,
            time_step=time_step
        )

        agents_observations = [obs] * n_agents
        value_critic = 0
        coma_advantage = 0
        score_value = 0
        for _ in range(n_samples):
            # Q value
            agents_actions = np.random.choice([0, 1], p=[1 - p_transmit, p_transmit], size=n_agents).tolist()
            value_critic_sample = coma_optimizer.get_value_critic(
                Transition(
                    agents_observations=agents_observations,
                    agents_actions=agents_actions,
                    agents_rewards=[],
                    agents_next_observations=[]
                )
            )
            value_critic += value_critic_sample

            # COMA Advantage and actor score
            coma_advantage_per_agent = []
            score_per_agent = []
            for idx_agent in range(n_agents):
                # COMA Advantage
                agents_actions_actor_action_0 = agents_actions[:idx_agent] + [0] + agents_actions[idx_agent + 1:]
                value_actor_action_0 = coma_optimizer.get_value_critic(
                    Transition(
                        agents_observations=agents_observations,
                        agents_actions=agents_actions_actor_action_0,
                        agents_rewards=[],
                        agents_next_observations=[]
                    )
                )
                agents_actions_actor_action_1 = agents_actions[:idx_agent] + [1] + agents_actions[idx_agent + 1:]
                value_actor_action_1 = coma_optimizer.get_value_critic(
                    Transition(
                        agents_observations=agents_observations,
                        agents_actions=agents_actions_actor_action_1,
                        agents_rewards=[],
                        agents_next_observations=[]
                    )
                )
                coma_advantage_agent = (
                        value_critic_sample -
                        ((1 - p_transmit) * value_actor_action_0) -
                        (p_transmit * value_actor_action_1)
                )
                coma_advantage_per_agent.append(coma_advantage_agent)

                # Score
                agent_action = agents_actions[idx_agent]
                p_action = (agent_action * p_transmit) + ((1 - agent_action) * (1 - p_transmit))
                score_per_agent.append(np.log(p_action) * coma_advantage_agent)

            score_value += np.mean(score_per_agent)
            coma_advantage += np.mean(coma_advantage_per_agent)

        value_critic = value_critic / n_samples
        coma_advantage = coma_advantage / n_samples
        score_value = score_value / n_samples

        data.append((ack, data_gen, n_packets_buffer, p_transmit, value_critic, coma_advantage, score_value))

    return pd.DataFrame(
        data,
        columns=["ack", "data_input", "n_packets_buffer", "p_transmit", "q_value", "coma_advantage", "score_value"]
    )
