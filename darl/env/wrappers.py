#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Env Wrapper"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

try:
    from future.builtins import ascii, filter, hex, map, oct, zip
except ImportError:
    pass
import sys

if sys.version_info.major > 2:
    xrange = range

import gym, random
import numpy as np

class AstypeEnv(gym.Wrapper):
    obs_dtype = None
    def __init__(self, env):
        """ Provide explicit type as numpy.float32 for gym.spaces.Box
        """
        super(AstypeEnv, self).__init__(env)
        self.obs_dtype = env.observation_space.dtype

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = obs.astype(self.obs_dtype)
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs = obs.astype(self.obs_dtype)
        return obs

class SingleAgentEnv(gym.Wrapper):
    def __init__(self, env):
        """ Make a gym.Env as a single agent Env.
        """
        super(SingleAgentEnv, self).__init__(env)
        self.observation_space = gym.spaces.Tuple([self.env.observation_space])
        self.action_space = gym.spaces.Tuple([self.env.action_space])

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.observation_space = gym.spaces.Tuple([self.env.observation_space])
        self.action_space = gym.spaces.Tuple([self.env.action_space])
        return [obs]
    
    def step(self, acts):
        assert hasattr(acts, '__iter__')
        obs, rwd, done, info = self.env.step(acts[0])
        return [obs], [rwd], [done], [info]

def make_robotics(env_id):
    env = gym.make(env_id)
    env = AstypeEnv(env)
    env = SingleAgentEnv(env)
    return env

if __name__ == '__main__':
    env = make_robotics('MountainCarContinuous-v0')
    print(env.observation_space)
    print(env.action_space)
    obs = env.reset()
    print(obs.dtype)

