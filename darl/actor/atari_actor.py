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
from darl.actor.actors import TrainActor
import numpy as np

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_bool("print_info", False, "Whether to print info")
flags.DEFINE_bool("save_img", False, "Whether to save images")

class AtariActor(TrainActor):
    def __init__(self, env, agt, **kwargs):
        super(AtariActor, self).__init__(env, agt, **kwargs)
        self.cur_step = 0
        self.sum_reward = [0.0] * len(self.agt)
        self.sum_discounted_reward = [0.0] * len(self.agt)
        self.cumulated_discount = 1.0

    def on_game_start(self, obs):
        #super(AtariActor, self).on_game_start(obs)
        pass

    def on_game_step(self, last_obs, a, obs, r, term, p ,v, info):
        super(AtariActor, self).on_game_step(last_obs, a, obs, r, term, p, v, info)
        if FLAGS.save_img and self.cur_step % 10 == 0:
            #print(obs.shape) # TODO: to be tested
            im = Image.fromarray(obs[0][0], mode='L')
            im = im.convert('RGB')
            im.save("step%05d.png" % self.cur_step)
        if FLAGS.print_info:
            if r != 0.0:
                print('rwd: ' + str(r))

    def on_game_end(self, info):
        for i,agt in enumerate(self.agt):
            if agt.client.push_endpoint and \
                    (info[i]['ale.lives'] == 0 or info[i]['sum_raw_step'] > 399800):
                agt.client.push_log("%d,%d,%e" % (info[i]['sum_raw_step'],
                    info[i]['sum_raw_reward'],self.sum_discounted_reward[i]))
                print("[%s] #game: %6d, #step: %6d" % \
                        (time.strftime("%Y-%m-%d %H:%M:%S"), self.game_idx, self.cur_step))
                self.env.unwrapped.reset()
                self.cur_step = 0
                self.game_idx += 1
                self.sum_reward = [0.0] * len(self.agt)
                self.sum_discounted_reward = [0.0] * len(self.agt)
                self.cumulated_discount = 1.0

