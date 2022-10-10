class EpsilonSchedule(object):
    def __init__(self, start_value, end_value, step):
        self._epsilon = start_value
        self.end_value = end_value
        self.step = step
        self.n_decimals = 7

    def get_epsilon(self, update_epsilon=True):
        if update_epsilon:
            self._epsilon = max(
                self.end_value,
                round(self._epsilon * self.step, self.n_decimals)
            )
        return self._epsilon
