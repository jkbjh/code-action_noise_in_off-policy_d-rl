import threading
import time
import numpy as np
import torch
from exploration_metrics.sampled import xurel_sampled as xurel_sampled_orig
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

_xurel_sampled_singlethreaded_lock = threading.Lock()


def xurel_sampled_singlethreaded(obs, low, high, **kwargs):
    with _xurel_sampled_singlethreaded_lock:
        return xurel_sampled_orig(obs, low, high, **kwargs)


def collect_qvaldata(model, size=5000):
    size = np.minimum(size, model.replay_buffer.size())
    replay_data = model.replay_buffer.sample(size, env=model._vec_normalize_env)
    q_values = torch.cat(
        model.critic(replay_data.observations, replay_data.actions), dim=1
    ).detach()
    uact = torch.as_tensor(
        np.random.uniform(
            model.env.action_space.low,
            model.env.action_space.high,
            size=replay_data.actions.shape,
        ).astype(dtype=np.float32)
    ).to(q_values.device)
    q_values_uniform_actions = torch.cat(
        model.critic(replay_data.observations, uact), dim=1
    ).detach()
    return q_values, q_values_uniform_actions


class CalculateRemainingTime(object):
    def __init__(
        self,
        total_timesteps,
        keep_steps=300,
        degree=5,
        mindegree=1,
        mintime=10,
        min_update_delay=10,
        alpha=0.1,
        train_dev=0.5,
    ):
        self.mintime = mintime
        self.min_update_delay = min_update_delay
        self.ests = {}
        self.est = None
        for d in range(mindegree, degree + 1):
            self.ests[d] = make_pipeline(
                PolynomialFeatures(d), Lasso(alpha=alpha, normalize=True, positive=True)
            )
        self.keep_steps = keep_steps
        self.degree = degree
        self.total_timesteps = total_timesteps
        self.train_dev = train_dev
        self.start_time = None
        self.last_time = None
        self.last_timesteps = None
        self.last_update_time = None
        self.timesteps = None
        self.est_degree = None
        self.reset(time.time(), 0)

    def reset(self, start_time, last_timesteps):
        self.start_time = start_time
        self.last_time = start_time
        self.last_update_time = start_time
        self.last_timesteps = last_timesteps
        self.timesteps = []

    def step(self, curtime, num_timesteps):
        remaining_time = None
        end_time = None
        if self.est:
            end_time = self.est.predict([[self.total_timesteps]]).item()
            remaining_time = end_time - curtime
        if (curtime - self.last_time) < self.mintime:
            return remaining_time, end_time
        self.timesteps.append((num_timesteps, curtime))
        self.last_time = curtime
        self.timesteps = self.timesteps[-self.keep_steps :]
        if (curtime - self.last_update_time) > self.min_update_delay:
            idc = np.arange(0, len(self.timesteps))
            np.random.shuffle(idc)
            i_train = idc[: int(len(idc) * self.train_dev)]
            i_dev = idc[int(len(idc) * self.train_dev) :]
            ts = np.array(self.timesteps)
            x = ts[i_train, 0]
            y = ts[i_train, 1]
            for est in self.ests.values():
                est.fit(x.reshape(-1, 1), y)
            scores = np.array(
                [
                    (est.score(ts[i_dev, 0:1], ts[i_dev, 1]), degree, est)
                    for degree, est in self.ests.items()
                ]
            )
            best_score, degree, est = scores[np.argmax(scores[:, 0])]
            self.est = est
            self.est_degree = degree
            self.last_update_time = curtime
            end_time = self.est.predict([[self.total_timesteps]]).item()
            remaining_time = end_time - curtime
        return remaining_time, end_time
