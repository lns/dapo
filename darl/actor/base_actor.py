#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function, unicode_literals)

try:
    from future.builtins import ascii, filter, hex, map, oct, zip
except ImportError:
    pass

from gym import spaces
import numpy as np
import signal

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        print("Receive signal %s. Exit gracefully." % str(signum))
        self.kill_now = True

class BaseActor(object):
    env = None
    agt = None # a list of agts

    def __init__(self, env, agt):
        self.env = env
        self.agt = agt
        assert hasattr(self.agt, '__iter__')
        assert isinstance(env.observation_space, spaces.Tuple)
        assert len(env.observation_space.spaces) == len(agt)
        assert isinstance(env.action_space, spaces.Tuple)
        assert len(env.action_space.spaces) == len(agt)

    def run(self):
        killer = GracefulKiller()
        while not killer.kill_now:
            obs = self.env.reset()
            for i,agt in enumerate(self.agt):
                agt.setup(self.env.observation_space[i],
                          self.env.action_space[i])
                agt.reset(obs[i])
            self.on_game_start(obs)
            while not killer.kill_now:
                last_obs = obs
                # a,p,v = agt.act(s)
                rets = [agt.act(s) for agt,s in zip(self.agt, obs)]
                a = [ret[0] for ret in rets]
                p = [ret[1] for ret in rets]
                v = [ret[2] for ret in rets]
                obs, rwd, term, info = self.env.step(a)
                self.on_game_step(last_obs, a, obs, rwd, term, p, v, info)
                if all(term) or killer.kill_now:
                    self.on_game_end(info)
                    break

    def on_game_start(self, obs):
        pass
    
    def on_game_step(self, last_obs, a, obs, r, term, p, v, info):
        pass

    def on_game_end(self, info):
        pass

