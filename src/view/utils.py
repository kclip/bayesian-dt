import pandas as pd
from typing import List


def smooth_columns(df: pd.DataFrame, colums: List[str], rolling_mean_window: int = None) -> pd.DataFrame:
    if rolling_mean_window is None:
        return df
    return df[colums].rolling(
        rolling_mean_window, min_periods=0
    ).mean().fillna(
        0, downcast="float64"
    )


def add_annotation(ax, label, pos_arrow, pos_text, arrow_kwargs=None, text_kwargs=None):
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path

    x_distance = pos_text[0] - pos_arrow[0]
    x_pos_bending = pos_arrow[0] + (0.75 * x_distance)
    pos_bending = (x_pos_bending, pos_text[1])
    _arrow_kwargs = arrow_kwargs or dict()
    arrow_patch = FancyArrowPatch(
        path=Path([pos_arrow, pos_bending, pos_text]),
        arrowstyle="<-",
        color="black",
        **_arrow_kwargs
    )
    ax.add_patch(arrow_patch)
    _text_kwargs = text_kwargs or dict()
    ax.text(
        pos_text[0], pos_text[1], f" {label}",
        ha="left", va="center",
        **_text_kwargs
    )
