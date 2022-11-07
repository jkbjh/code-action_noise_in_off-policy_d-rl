import numpy as np
from . import coverage
import joblib
from . import density

CPU_MODE = density.KNN is density.ScipyKNN


def xurel_sampled(obs, low, high, factor=2.0, samples=2000):
    ALL_IDCS = np.arange(len(obs))
    repetition = int(np.ceil(len(obs) / samples * factor))

    def selections():
        for _ in range(repetition):
            idcs = np.random.choice(ALL_IDCS, samples)
            selection = obs[idcs]
            yield selection

    if not CPU_MODE:
        results = [
            coverage.XUrel.coverage(low, high, selection) for selection in selections()
        ]
    else:
        jobs = (
            joblib.delayed(coverage.XUrel.coverage)(low, high, selection)
            for selection in selections()
        )
        results = joblib.Parallel()(jobs)
    return np.mean(results)
