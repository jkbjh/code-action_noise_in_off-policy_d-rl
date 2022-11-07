import joblib.memory
import os
from . import coverage

cache_path, _ = os.path.split(__file__)
memory = joblib.memory.Memory(cache_path)
VERSION = 2


class XUrel(coverage.XUrel):
    @staticmethod
    @memory.cache
    def estimate_k(dim, low=None, high=None, N=1000, M=5, steps=25, _version=VERSION):
        return coverage.XUrel.estimate_k(dim, low=low, high=high, N=N, M=M, steps=steps)
