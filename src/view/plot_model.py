import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from itertools import product
from scipy.stats import dirichlet, beta

from src.data_classes import Episode
from src.utils import encode_integers_tuple, decode_integers_tuple, max_likelihood_estimate, max_a_posteriori_estimate


def plot_dirichlet_data_generation(
        episode: Episode,
        probability_map_name: str,
        prior_dirichlet_concentration: float,
        prior_dirichlet_concentration_map: float = None,
        n_steps_per_plot: int = 100,
        true_transition_probability=None,
        x_axis_range: list = None,
        x_axis_step: float = 0.01,
        epsilon: float = 0.001
):
    _dirichlet_params_map_shift = (
          prior_dirichlet_concentration_map or prior_dirichlet_concentration
    ) + prior_dirichlet_concentration
    _logged_p_map = episode.history[0].digital_twin_info["data_generation_posterior_params"][probability_map_name]
    _adjacent_key = next(iter(_logged_p_map.keys()))
    _joint_key = next(
        k
        for v in _logged_p_map.values()
        for k in v.keys()
    )
    n_adjacent = len(decode_integers_tuple(_adjacent_key))
    n_joint = len(decode_integers_tuple(_joint_key))
    if n_adjacent > 0:
        all_adjacent_values = list(product(*[[0, 1] for _ in range(n_adjacent)]))
    else:
        all_adjacent_values = [""]
    all_joint_values = list(product(*[[0, 1] for _ in range(n_joint)]))
    _n_plots = len(all_adjacent_values)
    n_cols = len(all_joint_values)
    n_rows = len(all_adjacent_values)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[
            f"P({str(joint_values)} | {str(adjacent_values)})"
            for adjacent_values in all_adjacent_values
            for joint_values in all_joint_values
        ]
    )
    _x_axis_range = x_axis_range or [0, 1]
    if prior_dirichlet_concentration < 1:  # Avoid 0 values if prior is bellow 1
        _x_axis_range = [_x_axis_range[0] + epsilon, _x_axis_range[1] - epsilon]
    p = np.arange(_x_axis_range[0], _x_axis_range[1] + x_axis_step, x_axis_step)
    q = 1 - p
    beta_marginal_support = np.concatenate((p.reshape(1, -1), q.reshape(1, -1)), axis=0)

    # Add traces, one for each slider step
    idx_traces_per_step = []
    n_trace = 0
    episodes = episode.history[::n_steps_per_plot]
    for n_step, step in enumerate(episodes):
        idx_step_traces = []
        for n_row, adjacent_values in enumerate(all_adjacent_values):
            adjacent_values_key = encode_integers_tuple(adjacent_values, empty_tuple_encoding="")
            counter_given_adjacent = step.digital_twin_info[
                "data_generation_posterior_params"
            ][
                probability_map_name
            ][
                adjacent_values_key
            ]
            counter_given_adjacent_list = np.array([
                counter_given_adjacent[encode_integers_tuple(joint_values)]
                for joint_values in all_joint_values
            ])
            for n_col, joint_values in enumerate(all_joint_values):
                joint_values_key = encode_integers_tuple(joint_values)
                beta_marginal_params = [  # Marginal beta distribution of dirichlet parameters
                    counter_given_adjacent[joint_values_key],
                    sum(
                        count
                        for joint_values_counter, count in counter_given_adjacent.items()
                        if joint_values_counter != joint_values_key
                    )
                ]
                # Plot dirichlet pdf
                fig.add_trace(
                    go.Scatter(
                        visible=(n_step == len(episodes) - 1),
                        line=dict(color=px.colors.qualitative.T10[(n_row*n_cols + n_col) % 10], width=2),
                        name=f"Dir({beta_marginal_params})",
                        x=p,
                        y=dirichlet.pdf(beta_marginal_support, beta_marginal_params),
                        showlegend=True,
                    ),
                    row=n_row + 1,
                    col=n_col + 1
                )
                idx_step_traces.append(n_trace)
                n_trace += 1

                # Plot Maximum Likelihood estimate
                p_ml = max_likelihood_estimate(
                    posterior_dirichlet_params=counter_given_adjacent_list,
                    prior_dirichlet_concentration=prior_dirichlet_concentration
                )[all_joint_values.index(joint_values)]
                fig.add_trace(
                    go.Scatter(
                        visible=(n_step == len(episodes) - 1),
                        line=dict(color="green", width=1, simplify=True),
                        name="Maximum Likelihood",
                        marker=None,
                        x=[p_ml, p_ml],
                        y=[0, 10],
                        showlegend=((n_col == 0) and (n_row == 0))
                    ),
                    row=n_row + 1,
                    col=n_col + 1
                )
                idx_step_traces.append(n_trace)
                n_trace += 1

                # Plot Maximum A Posteriori estimate
                p_map = max_a_posteriori_estimate(
                    posterior_dirichlet_params=(_dirichlet_params_map_shift + counter_given_adjacent_list)
                )[all_joint_values.index(joint_values)]
                fig.add_trace(
                    go.Scatter(
                        visible=(n_step == len(episodes) - 1),
                        line=dict(color="red", width=1, simplify=True),
                        name="Maximum A Posteriori",
                        marker=None,
                        x=[p_map, p_map],
                        y=[0, 10],
                        showlegend=((n_col == 0) and (n_row == 0))
                    ),
                    row=n_row + 1,
                    col=n_col + 1
                )
                idx_step_traces.append(n_trace)
                n_trace += 1

                # Plot true probability value
                if true_transition_probability is not None:
                    fig.add_trace(
                        go.Scatter(
                            visible=(n_step == len(episodes) - 1),
                            line=dict(color="orange", width=1, simplify=True),
                            name="True probability",
                            marker=None,
                            x=[true_transition_probability[adjacent_values_key][joint_values_key]] * 2,
                            y=[0, 10],
                            showlegend=((n_col == 0) and (n_row == 0))
                        ),
                        row=n_row + 1,
                        col=n_col + 1
                    )
                    idx_step_traces.append(n_trace)
                    n_trace += 1

        # Keep record of traces pertaining to the same time step
        idx_traces_per_step.append(idx_step_traces)

    # Create and add slider
    slider_steps = []
    for n_step, idx_step_traces in enumerate(idx_traces_per_step):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},
                {"title": f"Dirichlet distributions for step: {n_step * n_steps_per_plot}"}
            ],  # layout attribute
        )
        for idx in idx_step_traces:  # Toggle every trace in step to "visible"
            step["args"][0]["visible"][idx] = True
        slider_steps.append(step)
    sliders = [
        dict(
            active=len(slider_steps) - 1,
            currentvalue={"prefix": "Number step: "},
            pad={"t": 50},
            steps=slider_steps
        )
    ]

    fig.update_layout(
        sliders=sliders,
        width=300*n_cols,
        height=200 + 200*n_rows,
    )

    fig.show()


def plot_dirichlet_mpr_channel(
        episode: Episode,
        prior_dirichlet_concentration: float,
        prior_dirichlet_concentration_map: float = None,
        n_steps_per_plot: int = 100,
        max_packets_transmitted: int = 3,
        true_mpr_matrix=None,
        epsilon: float = 0.001
):
    _dirichlet_params_map_shift = (
        prior_dirichlet_concentration_map or prior_dirichlet_concentration
    ) + prior_dirichlet_concentration

    n_rows = max_packets_transmitted
    n_cols = max_packets_transmitted + 1
    list_packets_transmitted = list(range(1, max_packets_transmitted + 1))
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[
            "" if n_packets_delivered > n_packets_transmitted else f"n_Tx={str(n_packets_transmitted)},n_Rx={str(n_packets_delivered)}"
            for n_packets_transmitted in list_packets_transmitted
            for n_packets_delivered in range(max_packets_transmitted + 1)
        ]
    )
    p_axis_step = 0.01
    _epsilon = 0
    if prior_dirichlet_concentration < 1:  # Avoid 0 values if prior is bellow 1
        _epsilon = epsilon
    p = np.arange(_epsilon, 1 + p_axis_step, p_axis_step)

    # Add traces, one for each slider step
    idx_traces_per_step = []
    n_trace = 0
    history_steps = episode.history[::n_steps_per_plot]
    for n_step, step in enumerate(history_steps):
        idx_step_traces = []
        for n_row_pos, n_packets_transmitted in enumerate(list_packets_transmitted):
            dir_params_keys = [
                encode_integers_tuple((n_packets_transmitted, n_packets_delivered))
                for n_packets_delivered in range(n_packets_transmitted + 1)
            ]
            dir_params = np.array([
                step.digital_twin_info["mpr_channel_posterior_params"].get(param_key)
                for param_key in dir_params_keys
            ])
            for n_col_pos, n_packets_delivered in enumerate(range(n_packets_transmitted + 1)):
                # Plot marginal distribution (beta) per n_packets_delivered
                marginal_param = dir_params[n_packets_delivered]
                other_param = sum(dir_params) - marginal_param
                beta_dist = beta(marginal_param, other_param)

                # Plot marginal pdf
                n_position = n_rows * n_row_pos + n_col_pos
                fig.add_trace(
                    go.Scatter(
                        visible=(n_step == len(history_steps) - 1),
                        line=dict(color=px.colors.qualitative.T10[n_position % 10], width=2),
                        name=f"Beta({marginal_param}, {other_param})",
                        x=p,
                        y=beta_dist.pdf(p),
                        showlegend=True,
                    ),
                    row=n_row_pos + 1,
                    col=n_col_pos + 1
                )
                idx_step_traces.append(n_trace)
                n_trace += 1

                # Plot Maximum Likelihood estimate
                p_ml = max_likelihood_estimate(
                    posterior_dirichlet_params=dir_params,
                    prior_dirichlet_concentration=prior_dirichlet_concentration
                )[n_packets_delivered]
                fig.add_trace(
                    go.Scatter(
                        visible=(n_step == len(history_steps) - 1),
                        line=dict(color="green", width=1, simplify=True),
                        name="Maximum Likelihood",
                        marker=None,
                        x=[p_ml, p_ml],
                        y=[0, 10],
                        showlegend=(n_position == 0)
                    ),
                    row=n_row_pos + 1,
                    col=n_col_pos + 1
                )
                idx_step_traces.append(n_trace)
                n_trace += 1

                # Plot Maximum A Posteriori estimate
                p_map = max_a_posteriori_estimate(
                    posterior_dirichlet_params=(_dirichlet_params_map_shift + dir_params)
                )[n_packets_delivered]
                fig.add_trace(
                    go.Scatter(
                        visible=(n_step == len(history_steps) - 1),
                        line=dict(color="red", width=1, simplify=True),
                        name="Maximum A Posteriori",
                        marker=None,
                        x=[p_map, p_map],
                        y=[0, 10],
                        showlegend=(n_position == 0)
                    ),
                    row=n_row_pos + 1,
                    col=n_col_pos + 1
                )
                idx_step_traces.append(n_trace)
                n_trace += 1

                if true_mpr_matrix is not None:
                    # Plot true probability value
                    fig.add_trace(
                        go.Scatter(
                            visible=(n_step == len(history_steps) - 1),
                            line=dict(color="orange", width=1, simplify=True),
                            name="True probability",
                            marker=None,
                            x=[true_mpr_matrix[n_packets_transmitted][n_packets_delivered]] * 2,
                            y=[0, 10],
                            showlegend=(n_position == 0)
                        ),
                        row=n_row_pos + 1,
                        col=n_col_pos + 1
                    )
                    idx_step_traces.append(n_trace)
                    n_trace += 1

        # Keep record of traces pertaining to the same time step
        idx_traces_per_step.append(idx_step_traces)

    # Create and add slider
    slider_steps = []
    for n_step, idx_step_traces in enumerate(idx_traces_per_step):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},
                {"title": f"Marginal dirichlet distributions for step: {n_step * n_steps_per_plot}"}
            ],  # layout attribute
        )
        for idx in idx_step_traces:  # Toggle every trace in step to "visible"
            step["args"][0]["visible"][idx] = True
        slider_steps.append(step)
    sliders = [
        dict(
            active=len(slider_steps) - 1,
            currentvalue={"prefix": "Number step: "},
            pad={"t": 50},
            steps=slider_steps
        )
    ]

    fig.update_layout(
        sliders=sliders,
        width=1000,
        height=800,
    )

    fig.show()