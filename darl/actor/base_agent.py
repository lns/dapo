#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function, unicode_literals)

try:
    from future.builtins import ascii, filter, hex, map, oct, zip
except ImportError:
    pass

import numpy as np
from darl.model.tf_model import TFModel

class BaseAgent(TFModel):
    observation_space = None
    action_space = None
    rnn = None

    def __init__(self, serial):
        self.deserialize(serial)
        for k,v in self.nodes.items():
            print("node['%s']: %s %s" % (k, str(v.dtype), str(v.shape)))
        if 'S' in self.nodes.keys():
            # drop first dimension (batchsize==1)
            self._states = np.zeros(self.nodes['S'].shape[1:],
                                    dtype=self.nodes['S'].dtype.as_numpy_dtype)
            self.rnn = True
        else:
            self.rnn = False

    def setup(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, obs=None):
        if self.rnn:
            self._states.fill(0)

    def _check_obs(self, obs):
        if not self.observation_space.contains(obs):
            print(self.observation_space)
            print(self.observation_space.low)
            print(self.observation_space.high)
            print(self.observation_space.shape)
            print(type(obs))
            print(obs.shape)
            print(obs)

    def act(self, obs):
        #self._check_obs(obs)
        assert self.observation_space.contains(obs)
        self.mutex.acquire()
        if self.rnn:
            a, p, self._states, v = self.sess.run( \
                    [self.nodes[k] for k in ['a','p','snew','v']],
                    feed_dict={ \
                            self.nodes['X']: np.asarray([obs]),
                            self.nodes['S']: np.asarray([self._states]),
                            self.nodes['M']: np.asarray([0])})
            self._states = self._states[0] # drop first dimension (batchsize==1)
        else:
            a, p, v = self.sess.run( \
                    [self.nodes[k] for k in ['a','p','v']],
                    feed_dict={ \
                            self.nodes['X']: np.asarray([obs])})
        self.mutex.release()
        # This check is omitted, as the action output by neural-network may
        # exceed the lower or higher bound of the box
        #assert self.action_space.contains(a[0])
        assert a[0].shape == self.action_space.shape
        return a[0], p[0], v[0]

    def step(self, obs):
        a, p, v = self.act(obs)
        return a

    @property
    def states(self):
        return self._states

