import numpy as np

try:
    from scipy.spatial import cKDTree as KDTree
except ImportError:
    from scipy.spatial import KDTree


class ScipyKNN(KDTree):
    def __init__(self, data, *args):
        n_points, n_features = data.shape
        assert n_points * n_features <= 700000
        super(ScipyKNN, self).__init__(data, *args)

    def query(self, data, k, **args):
        distances, _idcs = super(ScipyKNN, self).query(data, k, **args)
        if k == 1:
            return np.expand_dims(distances, 1), np.expand_dims(_idcs, 1)
        return distances, _idcs
