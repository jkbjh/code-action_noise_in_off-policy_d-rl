import numpy as np


class UniformBoxPrior(object):
    def __init__(self, low, high):
        assert np.all(high > low)
        self.low = low
        self.high = high
        self._log_density = np.log(1 / (high - low))

    def sample(self, num):
        return np.random.uniform(
            low=self.low, high=self.high, size=(num, self.high.shape[0])
        ).astype(np.float32)

    def log_prob(self, values):
        assert values.shape[1:] == self.high.shape
        tot_log_density = np.sum(self._log_density)
        within_limits = np.all(
            np.logical_and(self.low <= values, values <= self.high), axis=1
        )
        return np.where(within_limits, tot_log_density, 0)
