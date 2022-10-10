from typing import Tuple, Any

from src.data_classes import Observation


class Policy(object):
    def __init__(self):
        pass

    def action(self, observation: Observation) -> Tuple[int, Any]:
        # 1 -> Send, 0 -> Wait
        raise NotImplementedError()

    def reset(self):
        pass
