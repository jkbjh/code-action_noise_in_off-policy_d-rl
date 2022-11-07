import numpy as np
from . import entropy, priors, density
from scipy.optimize import fmin
import joblib


def get_normalized_low_high(low, high):
    assert len(low) == len(high)
    return -np.ones(len(low)), np.ones(len(high))


def normalize_data(low, high, data, mode="ignore"):
    if mode == "clip":
        data = np.clip(low, high, data)
    elif mode == "mod":
        data = np.mod(data - low, high - low + np.finfo(np.float32).eps) + low
        data = np.clip(low, high, data)
    halfspan = (high - low) / 2
    middle = (high + low) / 2.0
    normalized = (data - middle) / halfspan
    return normalized


def bin_coverage(observations, high, divisions=30, low=None):
    if low is None:
        low = -high
    clipped_obs = np.clip(observations, low, high)
    zero_one_scaled_obs = (clipped_obs - low) / (high - low)
    boxscaled_obs = (zero_one_scaled_obs * (divisions)).astype(np.int)
    filled_boxes = len({tuple(row) for row in boxscaled_obs})
    coverage = filled_boxes / (divisions ** high.shape[0])
    return coverage


def npt_bin_coverage_adjusted(low, high, observations, npt=5):
    dims = low.shape[0]
    n = observations.shape[0]
    divisions = int((len(observations) / npt) ** (1 / dims))
    coverage = bin_coverage(observations, high=high, low=low, divisions=divisions)
    boxes = (divisions) ** dims
    filled_boxes = coverage * boxes
    max_boxes = np.minimum(n, boxes)
    return np.minimum(filled_boxes / max_boxes, 1.0)


def bounding_box_mean_metric(data):
    return np.mean(np.ptp(data, axis=0))


def nuclear_norm_metric(data):
    return np.trace(np.cov(data.T))


def bi_nnr_relative_entropy_uniform_prior(low, high, data, k=None):
    U = priors.UniformBoxPrior(low, high)
    return -entropy.bi_nnr_relative_entropy(U.sample(len(data)), data, k=k)


class XUrel(object):
    @staticmethod
    def coverage(low, high, data, k=None):
        return bi_nnr_relative_entropy_uniform_prior(low, high, data, k=k)

    @staticmethod
    def from_volume_scale(k, volume_scale):
        return k * np.log(volume_scale)

    @classmethod
    def _fit_volume(cls_xurel, volumes, results):
        def objective(k):
            SSE = np.sum((results - cls_xurel.from_volume_scale(k, volumes)) ** 2)
            return SSE

        k_fit = fmin(objective, 0, disp=False)
        return k_fit

    @classmethod
    def _xurel_one_point(cls_xurel, low, high, scale, N=1000):
        half_span = (high - low) / 2.0
        middle = (high + low) / 2.0
        U = priors.UniformBoxPrior(
            middle - half_span * scale, middle + half_span * scale
        )
        return cls_xurel.coverage(low, high, U.sample(N))

    @classmethod
    def estimate_k(cls_xurel, dim, low=None, high=None, N=1000, M=5, steps=25):
        low = np.asarray([-1.0] * dim) if low is None else low
        high = np.asarray([1.0] * dim) if high is None else high
        scaled_volume = np.linspace(0.01, 1.0, steps)
        scales = scaled_volume ** (1 / dim)
        results = []
        for _ in range(M):
            for scale in scales:
                results.append(
                    joblib.delayed(cls_xurel._xurel_one_point)(low, high, scale, N=N)
                )
        results = joblib.Parallel()(results)
        results = np.reshape(results, (M, steps))
        k = cls_xurel._fit_volume(scaled_volume, np.mean(results, axis=0))
        return k
