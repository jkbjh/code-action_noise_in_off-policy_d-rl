import itertools
from typing import Iterable, List
from stable_baselines3.common.noise import VectorizedActionNoise
from . import dataset
from .dataset import EPISODE_T, SARSDI_T


def collect_rollouts(
    model,
    vecenv,
    max_steps=None,
    max_episodes=None,
    deterministic=False,
    action_noise: VectorizedActionNoise = None,
    incomplete: bool = False,
) -> dataset.Dataset:
    numenvs = vecenv.num_envs
    vecepisode_buffer: List[EPISODE_T] = [list() for i in range(numenvs)]
    allepisode_buffer: List[EPISODE_T] = []
    states = vecenv.reset()
    assert max_steps or max_episodes
    collected_steps = 0
    if max_steps is None or incomplete is False:
        allsteps: Iterable[int] = itertools.count(step=vecenv.num_envs)
    else:
        allsteps = range(0, max_steps, vecenv.num_envs)
    for _i in allsteps:
        actions, _ = model.predict(states)
        if action_noise:
            noise = action_noise()
            actions += noise
        nstates, rewards, dones, infos = vecenv.step(actions)
        if action_noise:
            action_noise.reset(dones)
        for j in range(numenvs):
            next_state = infos[j].pop("terminal_observation", nstates[j])
            sarsdi: SARSDI_T = (
                states[j],
                actions[j],
                rewards[j],
                next_state,
                dones[j],
                infos[j],
            )
            vecepisode_buffer[j].append(sarsdi)
            if dones[j]:
                allepisode_buffer.append(vecepisode_buffer[j])
                collected_steps += len(vecepisode_buffer[j])
                vecepisode_buffer[j] = list()
        if max_episodes and len(allepisode_buffer) > max_episodes:
            break
        if collected_steps >= max_steps:
            break
        states = nstates
    if incomplete:
        for j in range(numenvs):
            if len(vecepisode_buffer[j]) < 1:
                continue
            allepisode_buffer.append(vecepisode_buffer[j])
    ds = dataset.Dataset(episode_buffer=allepisode_buffer)
    return ds
