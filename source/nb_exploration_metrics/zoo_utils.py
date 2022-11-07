import pprint
import warnings
import json
import sb3_contrib.common.wrappers
import yaml
from stable_baselines3 import DDPG, SAC, TD3
from rlutils.detsac import DeterministicSAC
from stable_baselines3.common.utils import constant_fn
import numpy as np


def convert_learning_rate(learning_rate):
    if str(learning_rate).startswith("lin_"):
        lr = float(str(learning_rate)[len("lin_") :])
        print("lr>", lr)

        def linear_schedule(progress: float):
            return lr * progress

        return linear_schedule
    else:
        return constant_fn(float(learning_rate))


def load_algo_yaml(algorithm):
    if algorithm == "td3":
        ALGORITHM = TD3
        stable_baselines_algo_params = yaml.safe_load(
            open("stable-baselines3-zoo-td3.yml")
        )
    elif algorithm == "ddpg":
        ALGORITHM = DDPG
        stable_baselines_algo_params = yaml.safe_load(
            open("stable-baselines3-zoo-ddpg.yml")
        )
    elif algorithm == "sac":
        ALGORITHM = SAC
        stable_baselines_algo_params = yaml.safe_load(
            open("stable-baselines-zoo-sac.yml")
        )
    elif algorithm == "detsac":
        ALGORITHM = DeterministicSAC
        stable_baselines_algo_params = yaml.safe_load(
            open("stable-baselines-zoo-sac.yml")
        )
    else:
        raise AssertionError(f"{algorithm} is not supported!")
    return ALGORITHM, stable_baselines_algo_params


def get_params(yaml_data, ENVNAME, TOTAL_TIMESTEPS):
    if "PyBullet" in ENVNAME:
        ALT_ENVNAME = ENVNAME.replace("PyBullet", "Bullet")
    elif "Bullet" in ENVNAME:
        ALT_ENVNAME = ENVNAME.replace("Bullet", "PyBullet")
    else:
        ALT_ENVNAME = ""
    zoo_params = yaml_data.get(ENVNAME, None) or yaml_data.get(ALT_ENVNAME, None) or {}
    if "policy_kwargs" in zoo_params and isinstance(zoo_params["policy_kwargs"], str):
        zoo_params["policy_kwargs"] = eval(zoo_params["policy_kwargs"], {}, {})
    if "env_wrapper" in zoo_params:
        zoo_params["env_wrapper"] = eval(
            zoo_params["env_wrapper"], {"sb3_contrib": sb3_contrib}, {}
        )
    print("\nZoo params:")
    pprint.pprint(zoo_params)
    if "train_freq" in zoo_params:
        if isinstance(zoo_params["train_freq"], list):
            zoo_params["train_freq"] = tuple(zoo_params["train_freq"])
    copy_keys = {
        "policy",
        "batch_size",
        "buffer_size",
        "learning_starts",
        "learning_rate",
        "gamma",
        "gradient_steps",
        "policy_kwargs",
        "train_freq",
        "tau",
        "ent_coef",
        "env_wrapper",
        "noise_type",
        "noise_std",
    }
    ignore_keys = {"n_timesteps"}
    model_params = {}
    bad_keys = []
    warn_keys = {"use_sde"}
    for key in zoo_params:
        if key in copy_keys:
            model_params[key] = zoo_params[key]
        elif key in warn_keys:
            warnings.warn(f"The parameter {key} with value {zoo_params[key]} is used!")
            model_params[key] = zoo_params[key]
        elif key in ignore_keys:
            continue
        else:
            bad_keys.append(key)
    if bad_keys:
        raise AssertionError(f"there were unexpected keys: {bad_keys}")
    if "learning_rate" in model_params:
        model_params["learning_rate"] = convert_learning_rate(
            model_params["learning_rate"]
        )
    model_params.setdefault("policy", "MlpPolicy")
    return model_params, TOTAL_TIMESTEPS or zoo_params.get("n_timesteps", None)
