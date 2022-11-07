from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import datetime
import time
import functools
import exploration_metrics.coverage
import joblib
import numpy as np
import stable_baselines3.common.callbacks
from stable_baselines3.common.noise import VectorizedActionNoise
from . import evalutils, vecutils
from .evalutils import CalculateRemainingTime
from .evalutils import xurel_sampled_singlethreaded as xurel_sampled


class CBNTimes(stable_baselines3.common.callbacks.BaseCallback):
    def __init__(
        self,
        ntimes: Union[int, np.ndarray],
        callbacks: List[stable_baselines3.common.callbacks.BaseCallback],
    ):
        self.model: stable_baselines3.common.off_policy_algorithm.OffPolicyAlgorithm
        super(CBNTimes, self).__init__()
        self.callbacks = callbacks
        if isinstance(ntimes, int):
            ntimes = np.linspace(0.0, 1.0, ntimes)
        self.ntimes = ntimes
        print(f"ntimes: {ntimes}")
        self.index = 0
        self.last_time_trigger = -1.0
        self.remaining_points = iter(self.ntimes)
        self.next_time_trigger = next(self.remaining_points)

    def init_callback(self, model):
        super(CBNTimes, self).init_callback(model)
        for callback in self.callbacks:
            callback.init_callback(model)

    def _advance(self):
        try:
            self.next_time_trigger = next(self.remaining_points)
        except StopIteration:
            self.next_time_trigger = np.inf

    def _on_step(self) -> bool:
        percentage_done = self.num_timesteps / self.model._total_timesteps
        if percentage_done >= self.next_time_trigger:
            self._advance()
            for callback in self.callbacks:
                result = callback.on_step()
                if not result:
                    break
        return True


class CBDatasetBase(stable_baselines3.common.callbacks.BaseCallback):
    def __init__(self, *args, **kwargs):
        super(CBDatasetBase, self).__init__(*args, **kwargs)

    def _on_step(self, eval_dataset, *args, **kwargs):
        raise NotImplementedError()


class CBThreadParallel(stable_baselines3.common.callbacks.BaseCallback):
    def __init__(self, *callbacks):
        self.callbacks = callbacks
        super(CBThreadParallel, self).__init__()

    def init_callback(self, model):
        super(CBThreadParallel, self).init_callback(model)
        for callback in self.callbacks:
            callback.init_callback(model)

    def _on_step(self, *args, **kwargs) -> bool:
        results = [True]
        if self.callbacks:
            with joblib.parallel_backend(
                "threading", n_jobs=max(len(self.callbacks), 1)
            ):
                results = joblib.Parallel()(
                    joblib.delayed(callback.on_step)(*args, **kwargs)
                    for callback in self.callbacks
                )
        return all(results)


class CBDatasetLowHigh(CBDatasetBase):
    def __init__(self, callback, verbose=0):
        super(CBDatasetLowHigh, self).__init__(verbose)
        self.callback = callback
        self.min_state = None
        self.max_state = None
        self.min_action = None
        self.max_action = None
        self.min_return = None
        self.max_return = None
        self.min_eplength = None
        self.max_eplength = None

    def init_callback(self, model):
        super(CBDatasetLowHigh, self).init_callback(model)
        self.callback.init_callback(self.model)

    def _on_step(self, eval_dataset):
        actions = eval_dataset.collect_all_actions()
        states = eval_dataset.collect_all_states()
        returns = eval_dataset.collect_all_returns()
        eplengths = eval_dataset.collect_episode_lens()

        def minmax(values):
            return np.min(values, axis=0), np.max(values, axis=0)

        def update_minmax(minvals1, maxvals1, minvals2, maxvals2):
            assert minvals1 is not None or minvals2 is not None
            assert maxvals1 is not None or maxvals2 is not None
            if minvals1 is None:
                minvals = minvals2
            elif minvals2 is None:
                minvals = minvals1
            else:
                minvals = np.min([minvals1, minvals2], axis=0)
            if maxvals1 is None:
                maxvals = maxvals2
            elif maxvals2 is None:
                maxvals = maxvals1
            else:
                maxvals = np.max([maxvals1, maxvals2], axis=0)
            return minvals, maxvals

        actions_min, actions_max = minmax(actions)
        states_min, states_max = minmax(states)
        returns_min, returns_max = minmax(returns)
        eplengths_min, eplengths_max = minmax(eplengths)
        self.min_state, self.max_state = update_minmax(
            states_min, states_max, self.min_state, self.max_state
        )
        self.min_action, self.max_action = update_minmax(
            actions_min, actions_max, self.min_action, self.max_action
        )
        self.min_return, self.max_return = update_minmax(
            returns_min, returns_max, self.min_return, self.max_return
        )
        self.min_eplength, self.max_eplength = update_minmax(
            eplengths_min, eplengths_max, self.min_eplength, self.max_eplength
        )
        self.callback._on_step(eval_dataset)

    def to_dict(self) -> Dict:
        return dict(
            min_state=self.min_state,
            max_state=self.max_state,
            min_action=self.min_action,
            max_action=self.max_action,
            min_return=self.min_return,
            max_return=self.max_return,
            min_eplength=self.min_eplength,
            max_eplength=self.max_eplength,
        )

    def _merge_2_dicts(self, dict1: Dict, dict2: Dict) -> Dict:
        outdict = {}
        for key in dict1:
            if key.startswith("min"):
                outdict[key] = np.min([dict1[key], dict2[key]], axis=0).tolist()
            elif key.startswith("max"):
                outdict[key] = np.max([dict1[key], dict2[key]], axis=0).tolist()
        return outdict

    def reduce_to_dict(self, *cb_dataset_low_high: CBDatasetLowHigh) -> Dict:
        return functools.reduce(
            self._merge_2_dicts,
            [self.to_dict()] + [obj.to_dict() for obj in cb_dataset_low_high],
        )


class CBMultiEvalRollout(stable_baselines3.common.callbacks.BaseCallback):
    def __init__(
        self,
        eval_env,
        callbacks=None,
        max_steps=10000,
        max_episodes=100,
        verbose=0,
        deterministic=True,
        action_noise=None,
    ):
        super(CBMultiEvalRollout, self).__init__(verbose)
        if callbacks is None:
            callbacks = []
        self.callbacks = callbacks
        self.eval_env = eval_env
        self.max_steps = max_steps
        self.max_episodes = max_episodes
        self.deterministic = deterministic
        if action_noise is None:
            self.action_noise = None
        else:
            self.action_noise = VectorizedActionNoise(
                action_noise, self.eval_env.num_envs
            )

    def init_callback(self, model):
        super(CBMultiEvalRollout, self).init_callback(model)
        for callback in self.callbacks:
            callback.init_callback(self.model)

    def _on_step(self, *args, **kwargs) -> bool:
        eval_dataset = vecutils.collect_rollouts(
            self.model,
            self.eval_env,
            max_steps=self.max_steps,
            max_episodes=self.max_episodes,
            deterministic=self.deterministic,
            action_noise=self.action_noise,
        )
        for callback in self.callbacks:
            callback._on_step(eval_dataset)
        return True


class CriticBufferStatsCallback(stable_baselines3.common.callbacks.BaseCallback):
    def __init__(self, size=5000):
        self.logger: stable_baselines3.common.logger.Logger
        self.model: stable_baselines3.common.off_policy_algorithm.OffPolicyAlgorithm
        super(CriticBufferStatsCallback, self).__init__()
        self.size = size

    def _on_step(self, *args, **kwargs) -> bool:
        q_values, q_values_uniform_actions = evalutils.collect_qvaldata(
            self.model, size=self.size
        )
        self.logger.record("critic/qmin", q_values.min().item())
        self.logger.record("critic/qmax", q_values.max().item())
        self.logger.record("critic/qmean", q_values.mean().item())
        self.logger.record("critic/qstd", q_values.std().item())
        self.logger.record(
            "critic/buffer_size", np.float64(self.model.replay_buffer.size())
        )
        self.logger.record("critic/qua_min", q_values_uniform_actions.min().item())
        self.logger.record("critic/qua_max", q_values_uniform_actions.max().item())
        self.logger.record("critic/qua_mean", q_values_uniform_actions.mean().item())
        self.logger.record("critic/qua_std", q_values_uniform_actions.std().item())
        return True


class CBTimeRemaining(stable_baselines3.common.callbacks.BaseCallback):
    def __init__(
        self,
        keep_steps=2 * 60 * 60 // 5,
        degree=2,
        mintime=5.0,
        min_update_delay=60.0,
        alpha=1e-1,
        verbose=0,
    ):
        super(CBTimeRemaining, self).__init__(verbose)
        self.crt = None
        self.crt = CalculateRemainingTime(
            total_timesteps=1,
            keep_steps=keep_steps,
            degree=degree,
            mintime=mintime,
            min_update_delay=min_update_delay,
            alpha=alpha,
        )
        self.maxtime = datetime.datetime(5000, 1, 1).timestamp()
        self.max_dt = 10 ** 8

    def _on_step(self):
        curtime = time.time()
        self.crt.total_timesteps = self.model._total_timesteps
        remaining_time, end_time = self.crt.step(curtime, self.num_timesteps)
        if end_time is None:
            return True
        end_time = np.minimum(end_time, self.maxtime)
        remaining_time = np.minimum(remaining_time, self.max_dt)
        dt_end = datetime.datetime.fromtimestamp(end_time)
        self.logger.record("time/time_remaining", remaining_time)
        self.logger.record("time/time_end", dt_end)
        return True


class CBCoverageDS(CBDatasetBase):
    def __init__(
        self,
        low,
        high,
        verbose=0,
        samples=2000,
        prefix="me_",
        prefix_folder="eval",
        normalize_state_space=False,
    ):
        super(CBCoverageDS, self).__init__(verbose)
        self.low = low
        self.high = high
        self.ones = np.ones(low.shape, dtype=np.float32)
        self.samples = samples
        self.prefix = prefix
        if prefix_folder:
            self.prefix_folder = prefix_folder + "/"
        else:
            self.prefix_folder = ""
        self.normalize_state_space = normalize_state_space
        if normalize_state_space:
            (
                self.use_low,
                self.use_high,
            ) = exploration_metrics.coverage.get_normalized_low_high(
                self.low, self.high
            )
        else:
            self.use_low, self.use_high = low, high

    def _on_step(self, eval_dataset):
        returns = eval_dataset.collect_all_returns()
        rewards = eval_dataset.collect_all_rewards()
        mean = np.mean(returns)
        std = np.std(returns)
        q25, q50, q75 = np.quantile(returns, [0.25, 0.5, 0.75])
        prefix = self.prefix
        prefix_folder = self.prefix_folder
        self.logger.record(f"{prefix_folder}{prefix}q25_returns", q25)
        self.logger.record(f"{prefix_folder}{prefix}q50_returns", q50)
        self.logger.record(f"{prefix_folder}{prefix}q75_returns", q75)
        self.logger.record(f"{prefix_folder}{prefix}m_returns", mean.item())
        self.logger.record(f"{prefix_folder}{prefix}m_rewards", rewards.mean())
        self.logger.record(f"{prefix_folder}{prefix}s_returns", std.item())
        self.logger.record(f"{prefix_folder}{prefix}n_returns", len(returns))
        obs = eval_dataset.collect_all_states()
        if self.normalize_state_space:
            obs = exploration_metrics.coverage.normalize_data(self.low, self.high, obs)
        xurel_tanh = xurel_sampled(
            np.tanh(obs), low=-self.ones, high=self.ones, samples=self.samples
        )
        xurel_vals = xurel_sampled(
            obs, low=self.use_low, high=self.use_high, samples=self.samples
        )
        xbin = exploration_metrics.coverage.npt_bin_coverage_adjusted(
            self.use_low, self.use_high, obs, npt=1
        )
        self.logger.record(f"{prefix_folder}{prefix}xbin", xbin)
        self.logger.record(f"{prefix_folder}{prefix}xurel", xurel_vals)
        self.logger.record(f"{prefix_folder}{prefix}xurel_tanh", xurel_tanh)
        self.logger.record(
            f"{prefix_folder}{prefix}xbbm",
            exploration_metrics.coverage.bounding_box_mean_metric(obs),
        )
        self.logger.record(
            f"{prefix_folder}{prefix}xnn",
            exploration_metrics.coverage.nuclear_norm_metric(obs),
        )
        return True


class CBDumpLogger(stable_baselines3.common.callbacks.BaseCallback):
    def __init__(self):
        self.logger: stable_baselines3.common.logger.Logger
        super(CBDumpLogger, self).__init__()

    def _on_step(self) -> bool:
        self.logger.record(
            "time/total timesteps", self.num_timesteps, exclude="tensorboard"
        )
        self.logger.dump(self.num_timesteps)
        return True


class CBReplayBufferCoverages(stable_baselines3.common.callbacks.BaseCallback):
    def __init__(
        self,
        replay_buffer,
        low,
        high,
        samples=2000,
        verbose=0,
        normalize_state_space=False,
    ):
        self.logger: stable_baselines3.common.logger.Logger
        super(CBReplayBufferCoverages, self).__init__(verbose)
        self.replay_buffer = replay_buffer
        self.ones = np.ones(low.shape, dtype=np.float32)
        self.low = low
        self.high = high
        self.samples = samples
        self.normalize_state_space = normalize_state_space
        if normalize_state_space:
            (
                self.use_low,
                self.use_high,
            ) = exploration_metrics.coverage.get_normalized_low_high(
                self.low, self.high
            )
        else:
            self.use_low, self.use_high = low, high

    def _on_step(self, *args, **kwargs) -> bool:
        print("calculating xurel")
        buflen = self.replay_buffer.size()
        if buflen < self.samples:
            return True
        obs = self.replay_buffer.observations[:buflen, 0, :]
        if self.normalize_state_space:
            obs = exploration_metrics.coverage.normalize_data(self.low, self.high, obs)
        xurel_tanh = xurel_sampled(
            np.tanh(obs), low=-self.ones, high=self.ones, samples=self.samples
        )
        xurel_vals = xurel_sampled(
            obs, low=self.use_low, high=self.use_high, samples=self.samples
        )
        xbin = exploration_metrics.coverage.npt_bin_coverage_adjusted(
            self.use_low, self.use_high, obs, npt=1
        )
        self.logger.record("eval/xbin", xbin)
        self.logger.record("eval/xurel", xurel_vals)
        self.logger.record("eval/xurel_tanh", xurel_tanh)
        self.logger.record(
            "eval/xbbm", exploration_metrics.coverage.bounding_box_mean_metric(obs)
        )
        self.logger.record(
            "eval/xnn", exploration_metrics.coverage.nuclear_norm_metric(obs)
        )
        return True
