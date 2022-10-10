from src.data_classes import Episode
from src.view.plot import plot_episode


def plot_episode_from_logs(experiment_name: str, episode_name: str):
    episode = Episode.load_episode(experiment_name, episode_name)
    plot_episode(episode)
