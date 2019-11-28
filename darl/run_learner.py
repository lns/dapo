#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import (absolute_import, division, print_function, unicode_literals)

try:
    from future.builtins import ascii, filter, hex, map, oct, zip
except ImportError:
    pass
import sys

if sys.version_info.major > 2:
    xrange = range

from darl.learner.data_server import DataServer, DataServerPPO
from darl.model.policies import nature_cnn, mlp_net
from darl.env.suite import AutoMakeEnv
from darl.learner import ppo
from darl.learner.learner import Learner
from gym import spaces
import numpy as np
import os
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("devices", '0', "GPUs to use, seperated by comma.")
flags.DEFINE_integer("agent_index", 0, "Index of training agent")

def main(unused_argv):
    with open("config.txt", "w") as f:
        f.write(FLAGS.flags_into_string())
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.devices
    import tensorflow as tf

    env = AutoMakeEnv()
    env.reset()
    # Train the agent specified by FLAGS.agent_index
    ob_space = env.observation_space[FLAGS.agent_index]
    ac_space = env.action_space[FLAGS.agent_index]

    template = [np.zeros(ob_space.shape, ob_space.dtype),  # s
                np.zeros(ac_space.shape, ac_space.dtype),  # a
                np.zeros([], np.bool),     # term/mask
                np.zeros([], np.float32),  # r
                np.zeros([], np.float32),  # p
                np.zeros([], np.float32),  # v
                ]

    if FLAGS.suite == 'atari':
        ## Atari
        # env params
        base_net = nature_cnn
        # memoire params
        template[0] = np.zeros(ob_space.shape, np.uint8)
        if FLAGS.rnn:
            template.insert(3, np.zeros([2*FLAGS.nlstm], np.float32)) # s, a, m, h, r, p, v
    elif FLAGS.suite == 'robotics': # MountainCarContinuous
        # env params
        base_net = lambda x : mlp_net(x, nh=[256,256], activ=tf.nn.tanh)
        # memoire params
        if FLAGS.rnn:
            template.insert(3, np.zeros([2*FLAGS.nlstm], np.float32)) # s, a, m, h, r, p, v
    elif FLAGS.suite == 'arena':
        # env params
        base_net = lambda x : mlp_net(x, nh=[256,256], activ=tf.nn.tanh)
        # memoire params
        if FLAGS.rnn:
            template.insert(3, np.zeros([2*FLAGS.nlstm], np.float32)) # s, a, m, h, r, p, v
    else:
        raise RuntimeError("Unknown suite: %s" % str(FLAGS.suite))
    data_server = DataServerPPO(template=template, batch_size=FLAGS.batch_size,
                                discount_factor=[FLAGS.gamma], reward_coeff=[1.0])   

    print('Start Training')
    learner = Learner(data_server, ppo.Model, base_net=base_net, ob_space=ob_space, ac_space=ac_space)
    learner.run(tf.Session())

if __name__ == '__main__':
    app.run(main)
    os.kill(os.getpid(), 9)
