import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import List, Tuple

from src.utils import lineplot_ci
from src.data_classes import Episode
from src.view.utils import smooth_columns, add_annotation
from src.view.metrics import throughput_of_agents, get_joint_policy_action, get_buffer_info, get_return, \
    get_training_info, get_state_distribution, get_aac_info, get_loss, get_gradients_info, get_critic_training_info, \
    get_experiment_throughput, get_channel_collisions, experiment_throughput_mean_dev, \
    get_coma_info, get_coma_actors_info


def plot_episode(episode: Episode):
    max_throughput, agents_throughput, n_packets_sent = throughput_of_agents(episode)
    df_buffer = get_buffer_info([episode])
    joint_action = get_joint_policy_action(episode)
    length_episode = len(episode.history)
    plt.figure(figsize=(length_episode // 10, 12))
    plt.title(
        f"Episode plot for n_agents={episode.metadata.env_metadata.n_agents}, "
        f"n_packets_max={episode.metadata.env_metadata.n_packets_max}"
    )
    plt.plot(max_throughput)
    plt.plot(agents_throughput)
    plt.plot(df_buffer["sum_overflow"], ".-")
    plt.plot(joint_action, "--")
    plt.legend(
        ["Data generated", "Throughput", "Buffer Overflow", "Policy"],
        loc="lower left",
        bbox_to_anchor=(1, 0, 0, 0)
    )
    plt.show()


def plot_training(experiment_training_episodes: List[Episode]):
    df_training = get_training_info(experiment_training_episodes)
    div_episodes = len(experiment_training_episodes) // 3
    distributions = []
    for n_episodes in [1, div_episodes, 2 * div_episodes, len(experiment_training_episodes)]:
        distributions.append(
            (f"First {n_episodes} episodes", get_state_distribution(experiment_training_episodes[:n_episodes]))
        )

    plt.figure(figsize=(12, 12))
    plt.subplots_adjust(top=1.1)

    # Reward
    ax = plt.subplot(3, 2, 1)
    ax.set_title("Total average agent reward per episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.plot(df_training["episode"], df_training["total_reward_avg_agent"])

    # ACK
    ax = plt.subplot(3, 2, 2)
    ax.set_title("Number of packets sent per episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Number of packets")
    ax.plot(df_training["episode"], df_training["n_packets_sent"])

    # Overflow
    ax = plt.subplot(3, 2, 3)
    ax.set_title("Number of packets dropped (buffer overflow) per episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Number of packets")
    ax.plot(df_training["episode"], df_training["n_packets_dropped"])

    # Collisions
    ax = plt.subplot(3, 2, 4)
    ax.set_title("Number of channel collisions per episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Number of collisions")
    ax.plot(df_training["episode"], df_training["n_collisions"])

    # State_distribution (categorical)
    ax = plt.subplot(3, 2, 5)
    ax.set_title("Distribution of state categorical variables")
    ax.set_xlabel("Value")
    ax.set_ylabel("Probability")
    width = 0.3
    variables = ["ack", "data_input"]
    distributions_last_ep = distributions[-1][1]
    for i, cat in enumerate(variables):
        ax.bar(distributions_last_ep[cat].index + i * width, distributions_last_ep[cat], width)
    ax.legend(["ack", "data_input"])

    # Buffer state
    ax = plt.subplot(3, 2, 6)
    ax.set_title("Distribution of buffer state")
    ax.set_xlabel("Value")
    ax.set_ylabel("Probability")
    for label, distributions_ep in distributions:
        ax.plot(distributions_ep["buffer"].index, distributions_ep["buffer"], label=label)
    ax.legend()


def plot_validation_metrics(
        subplots: Tuple[plt.Figure, List[plt.Axes]],
        experiment_episodes: List[Episode],
        lineplot_kwargs: dict = None,
        rolling_mean_window: int = 100,
        n_packets_max: int = 10,
        average_packet_generation: float = None,
        max_throughput: float = 0.5,
        show_plot: bool = True,
):
    fig, axs = subplots
    axs = axs.flatten()
    lineplot_kwargs = lineplot_kwargs or dict()

    def _append_label(text: str):
        return {
            **lineplot_kwargs,
            "label": lineplot_kwargs.get("label", "") + " " + text
        }

    # Get metrics
    df_buffer = get_buffer_info(experiment_episodes)
    _cols = ["avg_buffer", "max_buffer"]
    df_buffer[_cols] = smooth_columns(df_buffer, _cols, rolling_mean_window=rolling_mean_window)
    df_throughput = get_experiment_throughput(experiment_episodes)
    _cols = ["throughput"]
    df_throughput[_cols] = smooth_columns(df_throughput, _cols, rolling_mean_window=rolling_mean_window)
    df_collision = get_channel_collisions(experiment_episodes)
    mean_throughput, dev_throughput = experiment_throughput_mean_dev(experiment_episodes)

    fig.set_size_inches(15, 10)

    # Buffer
    axs[0].set_ylim([0, n_packets_max + 1])
    axs[0].set_title(f"Average number of packets per agent buffer")
    lineplot_ci(ax=axs[0], df=df_buffer, x="step", y="avg_buffer", **_append_label("avg"))
    lineplot_ci(ax=axs[0], df=df_buffer, x="step", y="max_buffer", linestyle="dashed", **_append_label("max"))
    axs[0].legend()

    # Throughput
    axs[1].set_ylim([0, max_throughput])
    axs[1].set_title(f"Mean throughput")
    lineplot_ci(ax=axs[1], df=df_throughput, x="step", y="throughput",
                **_append_label(f"({mean_throughput:.3f} +- {dev_throughput:.3f})"))
    if average_packet_generation is not None:
        max_step = df_throughput["step"].max()
        axs[1].plot(
            [0, max_step], [average_packet_generation]*2,
            color="black", label="Avg packet generation", linestyle="dashed"
        )
    axs[1].legend()

    # Overflow
    axs[2].set_title(f"Overflow cumulated sum")
    lineplot_ci(ax=axs[2], df=df_buffer, x="step", y="cumsum_overflow", **lineplot_kwargs)
    axs[2].legend()

    # Collision
    axs[3].set_title(f"Collisions cumulated sum")
    lineplot_ci(ax=axs[3], df=df_collision, x="step", y="cumsum_collision", **lineplot_kwargs)
    axs[3].legend()

    if show_plot:
        plt.show()


def plot_validation_metrics_barplots(
        subplots: Tuple[plt.Figure, List[plt.Axes]],
        bar_pos: int,
        list_of_experiments: List[List[Episode]],
        n_steps_return: int,
        return_discount: float,
        barplot_kwargs: dict = None
):
    _std_coef = 1.96  # 95% confidence bound

    def _get_mean_std(s: pd.Series):
        mean = s.mean()
        std = None
        if len(df) > 1:
            std = _std_coef * s.std()
        return mean, std

    # Get dataframe with metrics of multiple experiments
    data = []
    for n_experiment, experiment_episodes in enumerate(list_of_experiments):
        # Get metrics
        df_buffer = get_buffer_info(experiment_episodes)
        df_reward = get_return(experiment_episodes, n_steps_return, return_discount)
        df_throughput = get_experiment_throughput(experiment_episodes)
        df_collision = get_channel_collisions(experiment_episodes)
        # Append mean data
        data.append({
            "n_experiment": n_experiment,
            "return": df_reward["return"].mean(),
            "throughput": df_throughput["throughput"].mean(),
            "avg_buffer": df_buffer["avg_buffer"].mean(),
            "max_buffer": df_buffer["max_buffer"].mean(),
            "collision": df_collision["sum_collision"].mean(),
            "overflow": df_buffer["sum_overflow"].mean(),
        })

    # Build Data Frame
    df = pd.DataFrame(data)

    # Plot
    fig, axs = subplots
    fig.set_size_inches(10, 5)
    fig.tight_layout()
    for ax in axs:
        ax.set_xticklabels([])
        ax.set_title("", fontsize=30, pad=15)
        ax.tick_params(axis='y', labelsize=20)

    # Throughput
    throughput_mean, throughput_std = _get_mean_std(df["throughput"])
    axs[0].bar(bar_pos, throughput_mean, yerr=throughput_std, ecolor="black", **barplot_kwargs)
    axs[0].title.set_text('Throughput')

    # Collision
    collision_mean, collision_std = _get_mean_std(df["collision"])
    axs[1].bar(bar_pos, collision_mean, yerr=collision_std, ecolor="black", **barplot_kwargs)
    axs[1].title.set_text('Collisions')

    # Buffer
    buffer_mean, buffer_std = _get_mean_std(df["avg_buffer"])
    axs[2].bar(bar_pos, buffer_mean, yerr=buffer_std, ecolor="black", **barplot_kwargs)
    axs[2].title.set_text('Buffer occupancy')

    # Overflow
    overflow_mean, overflow_std = _get_mean_std(df["overflow"])
    axs[3].bar(bar_pos, overflow_mean, yerr=overflow_std, ecolor="black", **barplot_kwargs)
    axs[3].title.set_text('Buffer overflow')


def plot_per_step_metrics(
        subplots: Tuple[plt.Figure, List[plt.Axes]],
        list_of_experiments: List[List[Episode]],
        n_steps_per_episode: int,
        n_steps_return: int,
        return_discount: float,
        lineplot_kwargs: dict = None,
        rolling_mean_window: int = 100,
        n_packets_max: int = 10,
        average_packet_generation: float = None,
        max_throughput: float = 1.0
):
    # Get dataframe with metrics of multiple experiments
    df_list = []
    for experiment_episodes in list_of_experiments:
        # Get metrics
        df_buffer = get_buffer_info(experiment_episodes)
        df_reward = get_return(experiment_episodes, n_steps_return, return_discount)
        df_throughput = get_experiment_throughput(experiment_episodes)
        df_collision = get_channel_collisions(experiment_episodes)
        # Merge and smooth
        _merge_keys = ["episode", "step"]
        df_temp = df_buffer.merge(
            df_reward, how="inner", on=_merge_keys
        ).merge(
            df_throughput, how="inner", on=_merge_keys
        ).merge(
            df_collision, how="inner", on=_merge_keys
        )
        _cols_to_smooth = [
            "return", "reward", "avg_buffer", "max_buffer", "sum_overflow", "sum_collision", "throughput"
        ]
        df_temp[_cols_to_smooth] = smooth_columns(df_temp, _cols_to_smooth, rolling_mean_window=rolling_mean_window)
        # Compute time step over all episodes
        df_temp["overall_step"] = df_temp["step"] + (n_steps_per_episode * df_temp["episode"])
        # Store in list
        df_list.append(df_temp)
    # Concat metrics of all experiments
    df = pd.concat(df_list, ignore_index=True)

    # Plot
    lineplot_kwargs = lineplot_kwargs or dict()
    fig, axs = subplots
    axs = axs.flatten()
    fig.set_size_inches(15, 40)

    # Reward
    axs[0].set_title(f"{n_steps_return}-steps return (smoothed {rolling_mean_window} steps)")
    lineplot_ci(ax=axs[0], df=df, x="overall_step", y="return", **lineplot_kwargs)
    axs[0].legend()

    # Buffer
    axs[1].set_ylim([0, n_packets_max + 1])
    axs[1].set_title(f"Average number of packets per agent buffer (smoothed {rolling_mean_window} steps)")
    lineplot_ci(ax=axs[1], df=df, x="overall_step", y="avg_buffer", **lineplot_kwargs)
    axs[1].legend()

    # Throughput
    axs[2].set_ylim([0, max_throughput])
    axs[2].set_title(f"Throughput (smoothed {rolling_mean_window} steps)")
    lineplot_ci(ax=axs[2], df=df, x="overall_step", y="throughput", **lineplot_kwargs)
    #     if average_packet_generation is not None:
    #         max_step = df_throughput["step"].max()
    #         axs[1].plot(
    #             [0, max_step], [average_packet_generation]*2,
    #             color="black", label="Avg packet generation", linestyle="dashed"
    #         )
    axs[2].legend()

    # Overflow
    axs[3].set_title(f"Overflow per time step (smoothed {rolling_mean_window} steps)")
    lineplot_ci(ax=axs[3], df=df, x="overall_step", y="sum_overflow", **lineplot_kwargs)
    axs[3].legend()

    # Collision
    axs[4].set_title(f"Collisions per time step (smoothed {rolling_mean_window} steps)")
    lineplot_ci(ax=axs[4], df=df, x="overall_step", y="sum_collision", **lineplot_kwargs)
    axs[4].legend()


def plot_validation_metrics_per_n_steps_model_learning(
        subplots: Tuple[plt.Figure, List[plt.Axes]],
        n_steps_range: List[int],
        list_of_experiments: List[List[Episode]],
        n_steps_return: int,
        return_discount: float,
        n_packets_max: int,
        lineplot_kwargs: dict = None,
        range_return: List[float] = None,
        range_throughput: List[float] = None,
        range_collision: List[float] = None,
        range_buffer: List[float] = None,
        range_overflow: List[float] = None,
        x_step_size: int = None,
        plot_return: bool = True,
        plot_throughput: bool = True,
        plot_buffer: bool = True,
        plot_buffer_max: bool = True,
        plot_collision: bool = True,
        plot_overflow: bool = True,
        annotate_x_pos=None,
        annotate_label=None
):
    def _annotate(ax, df, df_column, upside=True):  # Should be added after setting ylim
        if annotate_x_pos is not None and annotate_label is not None:
            y_pos_arrow = df[df["n_steps"] == annotate_x_pos][df_column].mean()
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            x_length = x_max - x_min
            y_length = y_max - y_min
            annotate_x_ratio = (annotate_x_pos - x_min) / x_length
            text_x_pos = x_max - (0.2 * x_length)
            text_y_pos = (
                (y_max - (annotate_x_ratio * y_length * (2/3)))
                if upside else
                (y_min + (annotate_x_ratio * y_length * (2/3)))
            )
            add_annotation(
                ax,
                annotate_label,
                (annotate_x_pos, y_pos_arrow),
                (text_x_pos, text_y_pos),
                arrow_kwargs=dict(
                    mutation_scale=15,
                    linewidth=1
                ),
                text_kwargs=dict(
                    fontsize=18
                )
            )

    def _append_label(text: str):
        return {
            **lineplot_kwargs,
            "label": lineplot_kwargs.get("label", "") + " " + text
        }

    # Get dataframe with metrics of multiple experiments
    df_list = []
    for n_steps, experiment_episodes in zip(n_steps_range, list_of_experiments):
        # Get metrics
        df_buffer = get_buffer_info(experiment_episodes) if (plot_buffer or plot_overflow) else None
        if df_buffer is not None: # Normalize buffer occupancy as percentage
            df_buffer[["avg_buffer", "max_buffer"]] = df_buffer[["avg_buffer", "max_buffer"]] * (100 / n_packets_max)
        df_reward = get_return(experiment_episodes, n_steps_return, return_discount) if plot_return else None
        df_throughput = get_experiment_throughput(experiment_episodes) if plot_throughput else None
        df_collision = get_channel_collisions(experiment_episodes) if plot_collision else None
        df_temp = None
        for df in [df_buffer, df_reward, df_throughput, df_collision]:
            if df is not None:
                if df_temp is None:
                    df_temp = df
                else:
                    df_temp = df_temp.merge(df, how="inner", on=["episode", "step"])
        # Mean metrics per episode
        df_gb = df_temp.groupby(
            "episode", as_index=False
        ).agg({
            **({"return": "mean"} if plot_return else {}),
            **({"throughput": "mean"} if plot_throughput else {}),
            **(
                {
                   "avg_buffer": "mean", "max_buffer": "mean", "norm_overflow": "mean"
                }
                if (plot_buffer or plot_overflow)
                else {}
            ),
            **({"norm_collision": "mean"} if plot_collision else {}),
        })
        df_gb["n_steps"] = n_steps
        df_list.append(df_gb)
    # Concat metrics of all experiments
    df = pd.concat(df_list, ignore_index=True)

    # Plot
    lineplot_kwargs = lineplot_kwargs or dict()
    fig, axs = subplots
    fig.tight_layout()
    axs = axs if hasattr(axs, "__iter__") else [axs]
    # axs = axs.flatten()
    fig.set_size_inches(10, len(axs) * 5)
    ax_counter = 0

    # Reward
    if plot_return:
        lineplot_ci(ax=axs[ax_counter], df=df, x="n_steps", y="return", **lineplot_kwargs)
        _range_return = range_return or axs[ax_counter].get_ylim()
        axs[ax_counter].set_ylim(_range_return)
        axs[ax_counter].set_ylabel("Return")
        _annotate(axs[ax_counter], df, "return", upside=False)
        ax_counter += 1

    # Buffer
    if plot_buffer:
        _range_buffer = range_buffer or [0, 110]
        axs[ax_counter].set_ylim(_range_buffer)
        _yticks = [
            n for n in range(0, 110, 10)
            if _range_buffer[0] <= n <= _range_buffer[1]
        ]
        axs[ax_counter].yaxis.set_ticks(_yticks)
        axs[ax_counter].set_yticklabels([f"{r}%" for r in _yticks])
        lineplot_ci(ax=axs[ax_counter], df=df, x="n_steps", y="avg_buffer", **_append_label("(avg)"))
        if plot_buffer_max:
            _max_buffer_kwargs = {
                **_append_label("(max)"),
                "linestyle": "dashed"
            }
            lineplot_ci(ax=axs[ax_counter], df=df, x="n_steps", y="max_buffer", **_max_buffer_kwargs)
        axs[ax_counter].set_ylabel("Buffer occupancy")
        _annotate(axs[ax_counter], df, "avg_buffer", upside=True)
        ax_counter += 1

    # Throughput
    if plot_throughput:
        lineplot_ci(ax=axs[ax_counter], df=df, x="n_steps", y="throughput", **lineplot_kwargs)
        _range_throughput = range_throughput or axs[ax_counter].get_ylim()
        axs[ax_counter].set_ylim(_range_throughput)
        axs[ax_counter].set_ylabel("Throughput")
        _annotate(axs[ax_counter], df, "throughput", upside=False)
        ax_counter += 1

    # Overflow
    if plot_overflow:
        _range_overflow = range_overflow or [0, 1.1]
        axs[ax_counter].set_ylim(_range_overflow)
        _yticks = [
            n/10 for n in range(11)
            if _range_overflow[0] <= n/10 <= _range_overflow[1]
        ]
        axs[ax_counter].yaxis.set_ticks(_yticks)
        lineplot_ci(ax=axs[ax_counter], df=df, x="n_steps", y="norm_overflow", **lineplot_kwargs)
        axs[ax_counter].set_ylabel("Buffer overflow probability")
        _annotate(axs[ax_counter], df, "norm_overflow", upside=True)
        ax_counter += 1

    # Collision
    if plot_collision:
        _range_collision = range_collision or [0, 1.1]
        axs[ax_counter].set_ylim(_range_collision)
        axs[ax_counter].yaxis.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        lineplot_ci(ax=axs[ax_counter], df=df, x="n_steps", y="norm_collision", **lineplot_kwargs)
        axs[ax_counter].set_ylabel("Collision probability")
        _annotate(axs[ax_counter], df, "norm_collision", upside=True)
        ax_counter += 1

    # Format axes
    for ax in axs:
        ax.tick_params(labelsize=20)
        ax.xaxis.label.set_size(20)
        ax.yaxis.label.set_size(20)
        ax.set_xlabel("Model-learning dataset size")
        x_start, x_end = ax.get_xlim()
        _x_step_size = x_step_size or ((x_end - x_start) // 10)
        ax.xaxis.set_ticks(np.arange(x_start+1, x_end, _x_step_size))


def plot_p_transmit_aac(
        ax: plt.Axes,
        experiment_name: str = None,
        df_aac: pd.DataFrame = None,
        y_min=-0.1, y_max=1.0,
        p_aloha=None,  # float or list
        p_column: str = "actor_p_transmit",
        y_label: str = "Probability of transmission"
):
    _df_aac = get_aac_info(experiment_name) if df_aac is None else df_aac
    legend = []
    ax.set_title(f"Policy transmission probability")
    ax.set_ylim([y_min, y_max])
    ax.set_xlabel('Number of packets in buffer')
    ax.set_ylabel(y_label)
    for (ack, data_in), dfg in _df_aac.groupby(["ack", "data_input"]):
        legend.append(f"ACK={ack}, data_in={data_in}")
        ax.plot(dfg["n_packets_buffer"], dfg[p_column])

    if p_aloha is not None:
        max_buffer = df_aac["n_packets_buffer"].max()
        p_aloha_list = [p_aloha] * (max_buffer + 1) if isinstance(p_aloha, float) else p_aloha
        range_buffer = range(0, max_buffer + 1)
        legend.append("ALOHA")
        ax.plot(range_buffer, p_aloha_list, color='black', linestyle='dashed')

    ax.legend(legend)


def plot_critic_value_aac(
        ax: plt.Axes,
        experiment_name: str = None,
        df_aac: pd.DataFrame = None
):
    _df_aac = get_aac_info(experiment_name) if df_aac is None else df_aac
    legend = []
    ax.set_title(f"Critic value function")
    ax.set_xlabel('Number of packets in buffer')
    ax.set_ylabel('Expected reward from state')
    for (ack, data_in), dfg in _df_aac.groupby(["ack", "data_input"]):
        legend.append(f"ACK={ack}, data_in={data_in}")
        ax.plot(dfg["n_packets_buffer"], dfg["critic_value"])
    ax.legend(legend)


def plot_training_actor_critic(
        experiment_training_episodes: List[Episode],
        return_discount: float,
        n_steps_return: int = 100,
        n_updates_smoothing: int = 10,
        actor_idx_gradient_info: int = 0,
        plot_avg_gradients: bool = True,
        plot_min_gradients: bool = False,
        plot_max_gradients: bool = False
):
    def _select_grad_cols(columns):
        return [
            col for col in columns if (col != "step") and (
                (plot_avg_gradients and col.startswith("mean")) or
                (plot_min_gradients and col.startswith("min")) or
                (plot_max_gradients and col.startswith("max"))
            )

        ]

    df_actor_training_info = get_loss(experiment_training_episodes, "loss_actor")
    cols = list(df_actor_training_info.columns)
    df_actor_training_info[cols] = smooth_columns(
        df_actor_training_info, cols, rolling_mean_window=n_updates_smoothing
    )

    df_gradients_actor = get_gradients_info(
        experiment_training_episodes, "gradients_info_actors", gradient_info_num=actor_idx_gradient_info
    )

    df_critic_training_info = get_critic_training_info(
        experiment_training_episodes,
        n_steps_return,
        return_discount
    )
    cols = list(df_critic_training_info.columns)
    df_critic_training_info_smoothed = smooth_columns(
        df_critic_training_info, cols, rolling_mean_window=n_updates_smoothing
    )
    df_gradients_critic = get_gradients_info(experiment_training_episodes, "gradients_info_critic")

    plt.figure(figsize=(50, 40))

    ax = plt.subplot(5, 1, 1)
    ax.plot(df_actor_training_info["step"], df_actor_training_info["loss"])
    ax.set_title(f"Actor training loss (Smoothed)")

    ax = plt.subplot(5, 1, 2)
    columns_grad_actor = _select_grad_cols(df_gradients_actor.columns)
    for col in columns_grad_actor:
        ax.plot(df_gradients_actor["step"], np.log(1 + df_gradients_actor[col]))
    ax.set_title(f"Actor NUM {actor_idx_gradient_info} gradients (Log Y-axis)")
    ax.legend(columns_grad_actor)

    ax = plt.subplot(5, 1, 3)
    ax.plot(df_critic_training_info_smoothed["step"], df_critic_training_info_smoothed["loss"])
    ax.set_title(f"Q value training loss (Smoothed)")

    ax = plt.subplot(5, 1, 4)
    columns_grad_critic = _select_grad_cols(df_gradients_critic.columns)
    for col in columns_grad_critic:
        ax.plot(df_gradients_critic["step"], np.log(1 + df_gradients_critic[col]))
    ax.set_title(f"Critic gradients (Log Y-axis)")
    ax.legend(columns_grad_critic)

    ax = plt.subplot(5, 1, 5)
    ax.plot(df_critic_training_info["step"], df_critic_training_info["estimated_target"], color="b")
    ax.plot(df_critic_training_info["step"], df_critic_training_info["estimated_value"], color="orange")
    ax.plot(df_critic_training_info["step"], df_critic_training_info["immediate_reward"], color="green")
    ax.plot(df_critic_training_info["step"], df_critic_training_info["discounted_n_steps_return"], "--", color="grey",
            alpha=0.2)
    ax.plot(df_critic_training_info_smoothed["step"], df_critic_training_info_smoothed["discounted_n_steps_return"],
            "--", color="purple")
    ax.legend([
        "Estimated target",
        "Estimated Q-value",
        "Immediate Reward",
        f"Discounted return ({n_steps_return} steps)",
        f"Discounted return ({n_steps_return} steps, Smoothed)",
    ])


def plot_coma_actors_critic(
        experiment_name: str,
        n_agents: int,
        samples_critic: int = 100,
        plot_individual_actors: bool = False,
        first_time_step: int = 0,
        frame_length: int = 1,
        y_max_actor: float = 1
):
    n_cols = frame_length
    n_rows = n_agents + 2 if plot_individual_actors else 2

    plt.figure(figsize=(7 * n_cols, 5 * n_rows))

    for slot in range(frame_length):
        p_aloha = 1 / n_agents

        df_coma_general = get_coma_info(
            experiment_name, samples_critic=samples_critic, time_step=(first_time_step + slot)
        )

        ax = plt.subplot(n_rows, n_cols, slot + 1)
        plot_p_transmit_aac(ax, df_aac=df_coma_general, y_max=y_max_actor, p_aloha=p_aloha)
        ax.set_title(f"Policies average transmission probability slot {slot+1}")

        ax = plt.subplot(n_rows, n_cols, n_cols + slot + 1)
        plot_critic_value_aac(ax, df_aac=df_coma_general)
        ax.set_title(f"Critic value function slot {slot+1}")

        if plot_individual_actors:
            df_list_coma_actors = get_coma_actors_info(experiment_name, time_step=(first_time_step + slot))
            for idx_actor, df_actor in enumerate(df_list_coma_actors):
                ax = plt.subplot(n_rows, n_cols, (idx_actor + 2) * n_cols + slot + 1)
                plot_p_transmit_aac(ax, df_aac=df_actor, y_max=y_max_actor, p_aloha=p_aloha)
                ax.set_title(f"Actor {idx_actor} transmission probability slot {slot+1}")


def plot_tdma_actors_intermediary_probabilities(
        experiment_name: str,
        n_agents: int,
        y_max_actor: float = 1,
        first_time_step: int = 0,
        frame_length: int = 1
):
    n_cols = 1 + frame_length
    n_rows = n_agents
    plt.figure(figsize=(6 * n_cols, 5 * n_rows))

    # Plot slot agnostic transmission probabilities
    df_list_coma_actors = get_coma_actors_info(experiment_name, time_step=first_time_step, is_tdma_actor=True)
    for idx_actor, df_actor in enumerate(df_list_coma_actors):
        ax = plt.subplot(n_rows, n_cols, n_cols * idx_actor + 1)
        plot_p_transmit_aac(
            ax,
            df_aac=df_actor,
            y_max=y_max_actor,
            p_aloha=None,
            p_column="slot_agnostic_p_transmit",
            y_label="Slot agnostic p_transmit"
        )
        ax.set_title(f"Actor {idx_actor + 1} slot agnostic transmission probability")

    for slot in range(frame_length):
        df_list_coma_actors = get_coma_actors_info(
            experiment_name, time_step=first_time_step + slot, is_tdma_actor=True
        )

        # Mean ALOHA per slot
        n_packets_max = df_list_coma_actors[0]["n_packets_buffer"].max()
        expected_number_of_agents_in_slot = np.array([0.0] * (n_packets_max+1))
        for df in df_list_coma_actors:
            expected_number_of_agents_in_slot += df.groupby(
                "n_packets_buffer", as_index=False
            ).agg({
                "p_slot": "mean"
            }).sort_values(
                "n_packets_buffer"
            )["p_slot"].values
        slot_occupation_dependent_p_aloha = [1/n for n in expected_number_of_agents_in_slot]

        # Plot slot selection
        for idx_actor, df_actor in enumerate(df_list_coma_actors):
            ax = plt.subplot(n_rows, n_cols, n_cols * idx_actor + slot + 2)
            plot_p_transmit_aac(
                ax,
                df_aac=df_actor,
                y_max=y_max_actor,
                p_aloha=slot_occupation_dependent_p_aloha,
                p_column="p_slot",
                y_label=f"Probability of selecting slot {slot+1}"
            )
            ax.set_title(f"Actor {idx_actor + 1} slot num. {slot+1} probability")
