#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Interface """

from __future__ import (absolute_import, division, print_function, unicode_literals)

try:
    from future.builtins import ascii, filter, hex, map, oct, zip
except ImportError:
    pass
import sys

if sys.version_info.major > 2:
    xrange = range

import gym, random
from arena.interfaces.interface import Interface
from arena.utils.spaces import NoneSpace

class ObsExtract(Interface):
    def __init__(self, interface, override=False):
        super(ObsExtract, self).__init__(interface)

    def reset(self, obs):
        super(ObsExtract, self).reset(obs)

    @property
    def observation_space(self):
        if isinstance(self.inter.observation_space, gym.spaces.Tuple):
            return self.inter.observation_space[0]
        else:
            return NoneSpace()

    def obs_trans(self, obs):
        obs = self.inter.obs_trans(obs)
        if len(obs) > 0:
            obs = obs[0]
        else:
            obs = None
        return obs

