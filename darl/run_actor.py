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

import time, os
import importlib
from absl import flags
from absl import app
from darl.actor.agent import Agent
# Atari
from darl.env.suite import AutoMakeEnv
from darl.actor.actors import TrainActor
from darl.actor.atari_actor import AtariActor

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # CPU

FLAGS = flags.FLAGS
flags.DEFINE_string("load_path", "", "Path for loading model")
flags.DEFINE_string("sub_ep", "", "Sub endpoint")
flags.DEFINE_string("req_ep", "", "Req endpoint")
flags.DEFINE_string("push_ep", "", "Push endpoint")
flags.DEFINE_integer("push_length", 256, "Length of push rollout. (0 for not push)")

def main(unused_argv):
    with open("config.txt", "w") as f:
        f.write(FLAGS.flags_into_string())
    print("load_path: '%s'" % str(FLAGS.load_path))

    env = AutoMakeEnv()

    if FLAGS.suite == 'atari':
        actor_cls = AtariActor
    elif FLAGS.suite == 'robotics':
        actor_cls = TrainActor
    elif FLAGS.suite == 'arena':
        actor_cls = TrainActor
    else:
        raise RuntimeError("Unknown suite: %s" % FLAGS.suite)

    n_agent = len(env.observation_space) # env.observation_space is a Tuple()
    agts = []

    serial = open(FLAGS.load_path, 'rb').read() if FLAGS.load_path else None
    for i in range(n_agent):
        if i==0:
            agt = Agent(serial=serial, sub_ep=FLAGS.sub_ep, req_ep=FLAGS.req_ep, push_ep=FLAGS.push_ep,
                        push_length=FLAGS.push_length)
        else:
            agt = Agent(serial=serial, sub_ep='', req_ep=FLAGS.req_ep, push_ep='', push_length=0)
        agts.append(agt)

    actor = actor_cls(env, agts)
    actor.run()

if __name__ == '__main__':
    app.run(main)

