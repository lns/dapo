#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function, unicode_literals)

try:
    from future.builtins import ascii, filter, hex, map, oct, zip
except ImportError:
    pass
import sys

if sys.version_info.major > 2:
    xrange = range

import os
import gym
from darl.env.interface import ObsExtract

from pysc2.env import sc2_env
from arena.env.sc2_base_env import SC2BaseEnv
from arena.wrappers.basic_env_wrapper import MergeUnits
from arena.wrappers.basic_env_wrapper import EpisodicLife
from arena.interfaces.sc2mini.units_order_int import UnitOrderInt
from arena.interfaces.sc2mini.act_wrappers import Discre4M2AInt
from arena.interfaces.sc2mini.act_wrappers import Discre8MnAInt
from arena.interfaces.sc2mini.act_wrappers import CombineActInt
from arena.interfaces.sc2mini.unit_attr_int import UnitAttrInt
from arena.utils.spaces import SC2RawObsSpace
from arena.utils.spaces import SC2RawActSpace
from arena.env.env_int_wrapper import EnvIntWrapper
from arena.agents.agt_int_wrapper import AgtIntWrapper
from arena.interfaces.interface import Interface
from arena.interfaces.raw_int import RawInt

class MultiPlayerTerm(gym.Wrapper):
    def __init__(self, env):
        """ Make term returned by step() as a list
        """
        super(MultiPlayerTerm, self).__init__(env)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs

    def step(self, acts):
        obs, rwd, done, info = self.env.step(acts)
        return obs, rwd, [done]*len(rwd), info

def get_mini_inter():
    inter = UnitOrderInt()
    inter = Discre4M2AInt(inter)
    inter = CombineActInt(inter)
    inter = UnitAttrInt(inter, override=True)
    inter = ObsExtract(inter)
    return inter

def make_arena(game_name):
    if game_name in ['ImmortalZealotNoReset']: #TODO: add more maps
        os.environ["http_proxy"]  = ""
        os.environ["https_proxy"] = ""
        inter_1 = get_mini_inter()
        inter_2 = get_mini_inter()
        players = [sc2_env.Agent(sc2_env.Race.terran), sc2_env.Agent(sc2_env.Race.terran)]
        env = SC2BaseEnv(players=players, agent_interface=None, map_name=game_name,
                screen_resolution=64, max_steps_per_episode=0, step_mul=1)
        env = MultiPlayerTerm(env)
        env = MergeUnits(env)
        env = EpisodicLife(env, max_lives=100, max_step=1000)
        env = EnvIntWrapper(env, [inter_1, inter_2])
    else:
        print("Unknown game: '%s'" % str(game_name))
        raise NotImplementedError
    return env

