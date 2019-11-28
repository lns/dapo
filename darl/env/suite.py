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

import gym, random
import numpy as np

from absl import flags, app
flags.DEFINE_string("game", "BreakoutNoFrameskip-v4", "name of the game environment")
flags.DEFINE_enum("suite", "atari", ['atari','robotics','arena'], "Suite of env wrappers")

def env_test(env, n_game=5):
    print(env.observation_space)
    print(env.observation_space.dtype)
    print(env.action_space)
    print(env.action_space.dtype)
    for game_idx in range(n_game):
        # start
        obs = env.reset()
        for i in range(1000):
            action = env.action_space.sample()
            obs, rwd, term, info = env.step(action)  # discrete
            if rwd != 0.0:
                print(rwd)
            if term:
                print(obs.shape)
                print(obs.dtype)
                print(i)
                break
        # close
        print("game_idx: %d" % game_idx)

def AutoMakeEnv():
    FLAGS = flags.FLAGS
    if FLAGS.suite == 'atari':
        from darl.env.atari_wrappers import make_atari
        env = make_atari(FLAGS.game)
    elif FLAGS.suite == 'robotics':
        from darl.env.wrappers import make_robotics
        env = make_robotics(FLAGS.game)
    elif FLAGS.suite == 'arena':
        from darl.env.arena_wrappers import make_arena
        env = make_arena(FLAGS.game)
    else:
        raise RuntimeError("Unknown suite: %s" % FLAGS.suite)
    env.reset()
    return env

def main(unused_argv):
    #env = make_atari('QbertNoFrameskip-v4')
    #env = make_robotics('Walker2d-v2')
    env = make_arena('ImmortalZealotNoReset')
    #import roboschool
    #env = make_robotics('RoboschoolHumanoid-v0') # TODO: not working, libpcre16.so.3 missing.
    env.reset()
    env_test(env, n_game=0)

if __name__ == '__main__':
    app.run(main)

