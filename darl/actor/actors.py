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

import time
from darl.actor.base_actor import BaseActor
import numpy as np

class TrainActor(BaseActor):
    game_idx = 0
    cur_step = 0
    sum_reward = None
    sum_discounted_reward = None
    gamma = 0.99

    def __init__(self, env, agt, **kwargs):
        super(TrainActor, self).__init__(env, agt, **kwargs)
        self.game_idx = 0
        self.gamma = 0.99

    def on_game_start(self, obs):
        self.cur_step = 0
        self.game_idx += 1
        self.sum_reward = [0.0]*len(self.agt)
        self.sum_discounted_reward = [0.0]*len(self.agt)
        self.cumulated_discount = 1.0

    def on_game_step(self, last_obs, a, obs, r, term, p, v, info):
        self.cumulated_discount *= self.gamma
        self.cur_step += 1
        for i,agt in enumerate(self.agt):
            if agt.client.push_length > 0:
                if agt.rnn:
                    # TODO: use term instead of mask
                    mask = True if self.cur_step == 0 else False
                    entry = [last_obs[i], a[i], mask, agt.states, r[i], p[i], v[i]]
                else:
                    entry = [last_obs[i], a[i], term[i], r[i], p[i], v[i]]
                agt.add_entry(entry, term[i])
            self.sum_reward[i] += r[i]
            self.sum_discounted_reward[i] += self.cumulated_discount * r[i]

    def on_game_end(self, info):
        for i,agt in enumerate(self.agt):
            if agt.client.push_endpoint:
                agt.client.push_log("%d,%e,%e" % \
                        (self.cur_step, self.sum_reward[i], self.sum_discounted_reward[i]))
            print("[%s] #game: %6d, #step: %6d" % \
                    (time.strftime("%Y-%m-%d %H:%M:%S"), self.game_idx, self.cur_step))

