from typing import Iterable, Tuple, List
from json import JSONEncoder

import numpy as np
from pandas import DataFrame
from matplotlib.pyplot import Axes


class JSONNumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def lineplot_ci(ax: Axes, df: DataFrame, x: str, y: str, color: str = "b", label=None, linestyle: str = "solid"):
    df = df.groupby(x)[y].agg(["mean", "sem"]).reset_index(drop=False)
    df["low"] = df["mean"] - 1.96 * df["sem"]
    df["high"] = df["mean"] + 1.96 * df["sem"]
    ax.plot(df[x], df["mean"], color=color, label=label, linestyle=linestyle)
    ax.fill_between(df[x], df["low"], df["high"], color=color, alpha=.20, linestyle=linestyle)
    ax.set_xlabel(x)
    ax.set_ylabel(y)


# Encode and decode tuples of integers into coma-separated strings

def encode_integers_tuple(integers_tuple: Iterable[int], empty_tuple_encoding: str = None) -> str:
    if len(integers_tuple) == 0:
        return empty_tuple_encoding
    return ",".join(map(str, integers_tuple))


def decode_integers_tuple(encoded_tuple: str) -> Tuple[int]:
    if len(encoded_tuple) == 0:
        return tuple()
    return tuple(int(c) for c in encoded_tuple.split(","))


# Compute maximum likelihood estimate from observations

def max_likelihood_estimate(
    posterior_dirichlet_params: List[float],
    prior_dirichlet_concentration: float
) -> List[float]:
    # Retrieve transition counts from dirichlet posterior params
    transition_counts = np.array(posterior_dirichlet_params) - prior_dirichlet_concentration
    total_counts = np.sum(transition_counts)
    if total_counts == 0:  # If no observation is made yet, default to uniform distribution
        n_components = len(posterior_dirichlet_params)
        return [(1 / n_components) for _ in range(n_components)]
    else:  # Return max likelihood estimate
        return list(transition_counts / total_counts)


def max_a_posteriori_estimate(posterior_dirichlet_params: List[float]) -> List[float]:
    if (np.array(posterior_dirichlet_params) <= 1).any():
        raise ValueError(
            f"Maximum A Posteriori is only defined for posterior Dirichlet distributions with all parameters strictly "
            f"above 1. Got paramters '{posterior_dirichlet_params}'"
        )
    map_numerators = np.array(posterior_dirichlet_params) - 1
    map_denominator = np.sum(map_numerators)
    return list(map_numerators / map_denominator)
