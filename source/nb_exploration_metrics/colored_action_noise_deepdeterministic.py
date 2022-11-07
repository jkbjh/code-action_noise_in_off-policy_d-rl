#!/usr/bin/env python
from __future__ import absolute_import
import argparse
import json
import pprint
import random
from math import ceil
import numpy as np
import stable_baselines3.common.callbacks
from stable_baselines3.common.callbacks import EveryNTimesteps
from rlutils.callbacks import (
    CBCoverageDS,
    CBDumpLogger,
    CBNTimes,
    CBDatasetLowHigh,
    CBMultiEvalRollout,
    CBReplayBufferCoverages,
    CBThreadParallel,
    CBTimeRemaining,
    CriticBufferStatsCallback,
)
from . import action_noises
from .action_noises import get_episode_length, make_env
from .zoo_utils import get_params, load_algo_yaml
from .argutil import empty_to_none

__VERSION__ = 5.1


def learning_experiment(
    ENVNAME,
    ACTION_NOISE_FACTORY,
    SEED,
    EP_LENGTH=None,
    TOTAL_TIMESTEPS=None,
    log_interval_eps=4,
    eval_interval=None,
    eval_interval_ep_factor=10,
    tblogdir="tblog",
    algorithm="td3",
    NUM_EVAL_ENVS=8,
    range_output=None,
    noise_scheduler=None,
    coverage_normalize_state_space=False,
    EVALUATE_NTIMES=100,
):
    assert eval_interval or eval_interval_ep_factor
    ALGORITHM, stable_baselines_algo_params = load_algo_yaml(algorithm)
    model_params, total_timesteps = get_params(
        stable_baselines_algo_params, ENVNAME, TOTAL_TIMESTEPS
    )
    with open("cn_env_params_arw2.json", "r") as fp:
        cn_env_params = json.load(fp)
    envwrapper = model_params.pop("env_wrapper", None)

    def wrapped_make_env(ENVNAME):
        env = make_env(ENVNAME)
        if envwrapper:
            return envwrapper(env)
        return env

    env = wrapped_make_env(ENVNAME)
    eval_env = wrapped_make_env(ENVNAME)
    multi_evalenv_1 = stable_baselines3.common.vec_env.DummyVecEnv(
        [lambda: wrapped_make_env(ENVNAME)] * NUM_EVAL_ENVS
    )
    multi_evalenv_2 = stable_baselines3.common.vec_env.DummyVecEnv(
        [lambda: wrapped_make_env(ENVNAME)] * NUM_EVAL_ENVS
    )
    ep_length = get_episode_length(env, EP_LENGTH)
    eval_interval = eval_interval or ceil(eval_interval_ep_factor * ep_length)
    print("EVAL Interval is:", eval_interval)
    print("Log interval is:", log_interval_eps)
    low = cn_env_params[ENVNAME]["low"]
    high = cn_env_params[ENVNAME]["high"]
    if envwrapper and (len(low) + 1 == len(env.observation_space.low)):
        high = np.concatenate([high, env.observation_space.high[-1:]])
        low = np.concatenate([low, env.observation_space.low[-1:]])
    orig_action_noise = ACTION_NOISE_FACTORY(env, model_params)
    if not noise_scheduler:
        action_noise = orig_action_noise
        noise_scheduler_cb = []
    else:
        action_noise = noise_scheduler.scheduled_noise(orig_action_noise)
        noise_scheduler_cb = [noise_scheduler]
    model_params.pop("noise_std", None)
    model_params.pop("noise_type", None)
    total_timesteps = total_timesteps or ep_length * 20
    print("\nActual Model params:")
    pprint.pprint(model_params)
    print("\n\n")
    model = ALGORITHM(
        env=env,
        action_noise=action_noise,
        verbose=True,
        tensorboard_log=tblogdir,
        seed=SEED,
        **model_params,
    )
    print(f"MODEL: {model!r}")
    xurel_callback_ = CBReplayBufferCoverages(
        model.replay_buffer,
        np.asarray(low),
        np.asarray(high),
        normalize_state_space=coverage_normalize_state_space,
    )
    minmax1 = CBDatasetLowHigh(
        CBCoverageDS(
            np.asarray(low),
            np.asarray(high),
            prefix_folder="meval",
            normalize_state_space=coverage_normalize_state_space,
        )
    )
    minmax2 = CBDatasetLowHigh(
        CBCoverageDS(
            np.asarray(low),
            np.asarray(high),
            prefix="an_",
            prefix_folder="eval_exploration",
            normalize_state_space=coverage_normalize_state_space,
        )
    )
    callbacks = noise_scheduler_cb + [
        CBNTimes(
            ntimes=np.linspace(0.0, 1.0, EVALUATE_NTIMES)[1:],
            callbacks=[
                CBThreadParallel(
                    CBMultiEvalRollout(multi_evalenv_1, callbacks=[minmax1]),
                    CBMultiEvalRollout(
                        multi_evalenv_2,
                        action_noise=action_noise,
                        deterministic=False,
                        callbacks=[minmax2],
                    ),
                    xurel_callback_,
                    CriticBufferStatsCallback(),
                ),
                CBDumpLogger(),
            ],
        ),
        EveryNTimesteps(n_steps=log_interval_eps, callback=CBTimeRemaining()),
    ]
    model.learn(
        tb_log_name=f"{ENVNAME.lower()}-{action_noise.NOISE_NAME}",
        total_timesteps=total_timesteps,
        log_interval=log_interval_eps,
        eval_env=eval_env,
        eval_freq=eval_interval,
        callback=callbacks,
    )
    if range_output:
        values = dict(envname=ENVNAME)
        values.update(minmax1.reduce_to_dict(minmax2))
        with open(range_output, "wt") as fp:
            json.dump(values, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--range-output", type=str, default="ranges.json")
    parser.add_argument("--version", type=float, default=__VERSION__)
    parser.add_argument("--algorithm", type=str, default="td3")
    parser.add_argument("--seed", type=int, default=random.randint(0, 2 ** 32 - 1))
    parser.add_argument("--coverage-normalize-state-space", action="store_true")
    parser.add_argument(
        "--log-interval-eps",
        type=float,
        default=4,
        help="how many episodes to wait before logging",
    )
    parser.add_argument("--mylabel", type=str, default="")
    parser.add_argument("--evaluate-ntimes", type=int, default=100)
    parser.add_argument(
        "--eval-interval-factor",
        type=float,
        default=4,
        help="how many steps (compared to the maximum episode length) to wait before evaluating",
    )
    parser.add_argument("--envname", type=str, required=True)
    if True:
        noise_group = parser.add_mutually_exclusive_group(required=True)
        noise_group.add_argument("--noise-color", type=float, default=None)
        noise_group.add_argument(
            "--noise-zoo", action="store_true", help="use the SB-Zoo params"
        )
        noise_group.add_argument(
            "--noise-gauss", action="store_true", help="use gaussian noise"
        )
        noise_group.add_argument("--noise-ou", action="store_true", help="use OU noise")
        parser.add_argument("--noise-scale", type=float, default=1.0)
        group = parser.add_argument_group("noise scheduler")
        group.add_argument("--noise-scheduler", type=str, default=None)
    else:
        parser.add_argument("--noise-color", type=float, required=True, default=0.0)
    parser.add_argument("--tbdir", type=str, default="tblog")
    parser.add_argument(
        "--ep-length",
        default=None,
        type=empty_to_none(int),
        help="Nr of steps per episode (if not deductible from environment)",
    )
    parser.add_argument(
        "--total-timesteps",
        default=None,
        type=empty_to_none(int),
        help="Total timesteps, otherwise taken from the config file",
    )
    args = parser.parse_args()
    assert args.version == __VERSION__
    assert args.algorithm in ("td3", "ddpg", "sac", "detsac")
    random.seed(args.seed)
    np.random.seed(args.seed)
    action_noise = action_noises.get_action_noise_factory(
        noise_color=args.noise_color,
        noise_ou=args.noise_ou,
        noise_gauss=args.noise_gauss,
        noise_zoo=args.noise_zoo,
        noise_scale=args.noise_scale,
    )
    if args.noise_scheduler in ("logistic_schedule", "linear_schedule"):
        noise_scheduler = action_noises.ActionNoiseSchedule(
            schedule=args.noise_scheduler, log_period=args.log_interval_eps
        )
    elif args.noise_scheduler == "epsilon_greedy":
        action_noise_factory_ = action_noise
        action_noise = lambda env, zooparams: action_noises.EpsilonGreedyActionNoise(
            action_noise_factory_(env, zooparams)
        )
        noise_scheduler = None
    else:
        noise_scheduler = None
    learning_experiment(
        args.envname,
        action_noise,
        args.seed,
        args.ep_length,
        TOTAL_TIMESTEPS=args.total_timesteps,
        eval_interval_ep_factor=args.eval_interval_factor,
        log_interval_eps=args.log_interval_eps,
        tblogdir=args.tbdir,
        algorithm=args.algorithm,
        range_output=args.range_output,
        noise_scheduler=noise_scheduler,
        coverage_normalize_state_space=args.coverage_normalize_state_space,
        EVALUATE_NTIMES=args.evaluate_ntimes,
    )
