import copy
import warnings
import colorednoise as cn
import numpy as np
import stable_baselines3.common.callbacks
from stable_baselines3.common.noise import ActionNoise


def make_env(name):
    import gym
    import pybullet_envs
    import pybulletgym
    import ex_s2r
    import gym_acrobot

    return gym.make(name)


def make_actions(low, high, samples, color, scale=1.0):
    assert len(low.shape) == 1
    (action_dim,) = low.shape
    actions = (
        np.stack(
            [cn.powerlaw_psd_gaussian(color, samples) for i in range(action_dim)], -1
        )
        * scale
    )
    middle = (low + high) / 2.0
    span = high - low
    actions = np.clip((actions) * span / 2.0 + middle, low, high)
    return actions


class EnvActionNoise(ActionNoise):
    def __init__(self, envspec, ep_length, scale=1.0):
        if isinstance(envspec, str):
            env = make_env(envspec)
        else:
            env = envspec
        self.envspec = str(envspec)
        self.ep_length = get_episode_length(env, ep_length)
        self.scale = scale
        self.low = env.action_space.low
        self.high = env.action_space.high


class OUActionNoise(EnvActionNoise):
    def __init__(self, envspec, ep_length=None, scale=1.0):
        super(OUActionNoise, self).__init__(envspec, ep_length, scale=scale)
        self.ou_gen = None

    def reset(self):
        self.ou_gen = ornstein_uhlenbeck_gen(self.low, self.high, sigma=self.scale)

    def __call__(self):
        return next(self.ou_gen)


class GaussActionNoise(EnvActionNoise):
    def __init__(self, envspec, ep_length=None, scale=1.0):
        super(GaussActionNoise, self).__init__(envspec, ep_length, scale=scale)
        self.rescaler = action_rescaler(self.low, self.high)

    def __call__(self):
        return self.rescaler(np.random.normal(scale=self.scale, size=self.low.shape))


class ColoredActionNoise(EnvActionNoise):
    def __init__(self, envspec, noise_color=0.0, ep_length=None, scale=1.0):
        super(ColoredActionNoise, self).__init__(envspec, ep_length, scale=scale)
        self.action_iter = None
        self.noise_color = noise_color
        self.reset()
        print("ep_length:", self.ep_length)

    def reset(self):
        self.actions = make_actions(
            self.low, self.high, self.ep_length + 10, self.noise_color, scale=self.scale
        )
        self.action_iter = iter(self.actions)

    def __call__(self):
        try:
            return next(self.action_iter)
        except StopIteration:
            warnings.warn("ohoh, colored noise ran out of actions"),
            self.reset()
            return next(self.action_iter)


def ornstein_uhlenbeck_unscaled(
    mu: np.ndarray = None,
    sigma: float = 1.0,
    dim: int = None,
    theta: float = 0.15,
    dt: float = 1e-2,
):
    if mu is None:
        if not dim:
            raise ValueError("Either mu or dimensionality parameter dim must be set!")
        mu = np.zeros(dim)
    last_noise = np.zeros_like(mu)
    while True:
        noise = (
            last_noise
            + theta * (mu - last_noise) * dt
            + sigma * np.sqrt(dt) * np.random.normal(size=mu.shape)
        )
        last_noise = noise
        yield noise


def action_rescaler(low, high):
    halfspan = (high - low) / 2.0
    middle = (high + low) / 2.0

    def rescaler(action):
        return np.clip((action + middle) * halfspan, low, high)

    return rescaler


def ornstein_uhlenbeck_gen(
    low: np.ndarray,
    high: np.ndarray,
    mu: np.ndarray = None,
    sigma: float = 1.0,
    theta: float = 0.15,
    dt: float = 1e-2,
):
    rescaler = action_rescaler(low, high)
    middle = (low + high) / 2.0
    if mu is None:
        mu = middle
    noise_gen = ornstein_uhlenbeck_unscaled(mu=mu, sigma=sigma, theta=theta, dt=dt)
    for action in noise_gen:
        yield rescaler(action)


def get_episode_length(env, episode_length=None):
    if episode_length is None:
        if hasattr(env, "get_attr"):
            (episode_length,) = env.get_attr("_max_steps", 0)
        episode_length = env.__dict__.get("_max_steps", episode_length)
        episode_length = env.__dict__.get("_max_episode_steps", episode_length)
        if episode_length is None and hasattr(env, "env"):
            episode_length = get_episode_length(env.env, episode_length)
        if episode_length is None:
            raise AssertionError()
    return episode_length


def get_action_noise_factory(
    noise_color=None, noise_ou=None, noise_gauss=None, noise_zoo=None, noise_scale=None
):
    if (
        noise_color is not None or noise_ou is not None or noise_gauss is not None
    ) and noise_scale is None:
        raise ValueError(
            "noise_color, noise_ou, noise_gauss needs noise_scale to set, which it is not."
        )
    if not (noise_color is not None or noise_ou or noise_gauss or noise_zoo):
        raise ValueError(
            "one of noise_color, noise_ou, noise_gauss or noise_zoo needs to be set."
        )

    def _color_action_noise(env, zoo_params=None):
        action_noise = ColoredActionNoise(
            env, noise_color=noise_color, scale=noise_scale
        )
        action_noise.NOISE_NAME = noise_color
        return action_noise

    def _ou_action_noise(env, zoo_params=None):
        action_noise = OUActionNoise(env, scale=noise_scale)
        action_noise.NOISE_NAME = "OU"
        return action_noise

    def _gauss_action_noise(env, zoo_params=None):
        action_noise = GaussActionNoise(env, scale=noise_scale)
        action_noise.NOISE_NAME = "Gauss"
        return action_noise

    def _zoo_action_noise(env, zoo_params=None):
        if "noise_std" not in zoo_params or "noise_type" not in zoo_params:
            print("WARNING! no action noise defined setting gaussian-zero.")
            action_noise = GaussActionNoise(env, scale=0.0)
            action_noise.NOISE_NAME = "ZOO_Null"
        else:
            scale = zoo_params["noise_std"]
            noise_type = zoo_params["noise_type"]
            if noise_type == "normal":
                action_noise = GaussActionNoise(env, scale=scale)
                action_noise.NOISE_NAME = "ZOO_Gauss"
            elif noise_type == "ornstein-uhlenbeck":
                action_noise = OUActionNoise(env, scale=scale)
                action_noise.NOISE_NAME = "ZOO_OU"
            else:
                raise NotImplementedError(
                    f"noise_type {noise_type} not implemented. uauauaua"
                )
        return action_noise

    if noise_color is not None:
        action_noise = _color_action_noise
    elif noise_ou is not None:
        action_noise = _ou_action_noise
    elif noise_gauss is not None:
        action_noise = _gauss_action_noise
    elif noise_zoo is not None:
        action_noise = _zoo_action_noise
    return action_noise


class ProxiedActionNoise(ActionNoise):
    def __init__(self, action_noise):
        self.action_noise = action_noise

    def reset(self):
        return self.action_noise.reset()

    def __call__(self):
        return self.action_noise()

    def __deepcopy__(self, memo):
        newone = type(self)(None, None)
        newone.__dict__.update(self.__dict__)
        newone.action_noise = copy.deepcopy(self.action_noise, memo)
        return newone

    def __getattr__(self, attr):
        return getattr(self.action_noise, attr)


class EpsilonGreedyActionNoise(ProxiedActionNoise):
    def __init__(self, action_noise, eps=0.5):
        super(EpsilonGreedyActionNoise, self).__init__(action_noise)
        self.eps = eps
        self._eps_gen = self._setup_greedy_gen(self.eps)

    def __deepcopy__(self, memo):
        newone = super(EpsilonGreedyActionNoise, self).__deepcopy__(memo)
        newone._eps_gen = self._setup_greedy_gen(self.eps)
        return newone

    @staticmethod
    def _setup_greedy_gen(eps, batch_size=512):
        while True:
            for weight in np.random.binomial(1, eps, size=(batch_size,)).astype(
                np.float32
            ):
                yield weight

    def __call__(self):
        epsilon = next(self._eps_gen)
        action = self.action_noise()
        return action * epsilon


class ScheduledActionNoise(ProxiedActionNoise):
    def __init__(self, action_noise_schedule, action_noise):
        super(ScheduledActionNoise, self).__init__(action_noise)
        self.action_noise_schedule = action_noise_schedule

    def __call__(self):
        return self.action_noise_schedule.scale * self.action_noise()


class ActionNoiseSchedule(stable_baselines3.common.callbacks.BaseCallback):
    def __init__(self, schedule="logistic_schedule", log_period=100):
        super(ActionNoiseSchedule, self).__init__()
        self.total_timesteps = None
        self.percent_done = 0.0
        self.scale = 1.0
        self.scheduler = (
            getattr(self, schedule) if isinstance(schedule, str) else schedule
        )
        self.last_output_step = 0.0
        self.log_period = log_period

    @staticmethod
    def logistic_schedule(x, k=11, x0=0.5, start=1.0, end=0.0):
        l = 1
        change = start - end
        values = start - l / (1 + np.exp(-k * (x - x0))) * change
        return values

    @staticmethod
    def linear_schedule(x, start=1.0, end=0.0):
        change = start - end
        return start - x * change

    def scheduled_noise(self, action_noise):
        return ScheduledActionNoise(self, action_noise)

    def _on_step(self):
        self.percent_done = self.num_timesteps / self.model._total_timesteps
        self.scale = self.scheduler(self.percent_done)
        if (self.num_timesteps - self.last_output_step) >= self.log_period:
            self.logger.record("noise_scheduler_scale", self.scale)
            self.last_output_step = self.num_timesteps
        return True
