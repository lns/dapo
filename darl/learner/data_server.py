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

from darl.memoire import ReplayMemoryServer, Bind, Conn
from darl.utils.tf_data import TFData
from darl.utils.device import get_visible_devices
from threading import Thread
from collections import namedtuple
import time
import numpy as np
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("port", 5560, "3 successive ports are used for sub, req, and push.")
flags.DEFINE_float("priority_exponent", 0.0, "priority exponent for sampling")
flags.DEFINE_integer("max_step", 65536, "Max number of steps in a memory slot.")
flags.DEFINE_integer("max_episode", 2, "Max number of episodes in a memory slot (0 for infty).")
flags.DEFINE_integer("n_slot", 192, "Number of memory slot. Should be greater than number of actors.")
flags.DEFINE_integer("pull_buf_size", 8388608, "Pull buffer size (in bytes)")
flags.DEFINE_string("logfile_path", "out/logfile", "Path of log file")
flags.DEFINE_float("mix_lambda", 0.9, "Lambda used in TD(lambda)")
flags.DEFINE_float("priority_decay", 1.0, "Decay priority for later states")
flags.DEFINE_float("traceback_threshold", 1e-3, "Stopping criteria for GAE update")
flags.DEFINE_integer("rollout_len", 1, "Length of rollout (e.g. for RNN)")
flags.DEFINE_bool("do_padding", False, "Padding initial state (e.g. frame stack)")
flags.DEFINE_integer("n_pull_worker", 4, "Number of Pull workers")
flags.DEFINE_integer("n_data_worker", 16, "Number of prefetch workers")

class DataServer(object):
    def __init__(self,
                 template,
                 discount_factor=[0.99],
                 reward_coeff=[1.0]):
        pub_port = FLAGS.port
        rep_port = FLAGS.port + 1
        pull_port = FLAGS.port + 2
        self.template = template
        self.server = ReplayMemoryServer(tuple(template),
                                         max_step=FLAGS.max_step,
                                         n_slot=FLAGS.n_slot)
        self.server.pub_endpoint = "tcp://*:%d" % pub_port
        self.server.set_logfile(FLAGS.logfile_path, 'w')
        self.server.pull_buf_size = FLAGS.pull_buf_size
        rem = self.server.rem
        rem.max_episode         = FLAGS.max_episode
        rem.priority_exponent   = FLAGS.priority_exponent
        rem.mix_lambda          = FLAGS.mix_lambda
        rem.priority_decay      = FLAGS.priority_decay
        rem.traceback_threshold = FLAGS.traceback_threshold
        rem.rollout_len         = FLAGS.rollout_len
        rem.do_padding          = FLAGS.do_padding
        rem.discount_factor     = discount_factor
        rem.reward_coeff        = reward_coeff
        self.server.print_info()
        self.rem = rem

        threads = []
        threads.append(Thread(target=self.server.rep_worker_main,
                              args=("tcp://*:%d" % rep_port, Bind)))
        threads.append(Thread(target=self.server.rep_proxy_main,
                              args=("tcp://*:%d" % pull_port, Bind,
                                    "inproc://pull_workers", Bind,
                                    16*FLAGS.n_pull_worker)))
        for i in range(FLAGS.n_pull_worker):
            threads.append(Thread(target=self.server.pull_worker_main,
                                  args=("inproc://pull_workers", Conn)))
        for th in threads:
            th.start()

    def __del__(self):
        self.server.close()

    def pub(self, msg):
        self.server.pub_bytes('darl.model', msg)
        print("Pub model!")


def flatten(X):
    shape = X.shape
    assert len(shape) > 1
    new_shape = [shape[0]*shape[1]] + list(shape[2:])
    return(X.reshape(new_shape))


class DataServerPPO(DataServer):
    """ DataServer for PPO """
    struct = None # namedtuple
    dtypes = None # list of dtypes
    shapes = None # list of shapes
    generator_func = None
    input_data = {} # device.name -> (prefetched) tensors as namedtuple 'InputData'

    def __init__(self, batch_size, **kwargs):
        super(DataServerPPO, self).__init__(**kwargs)
        names = ['X', 'A', 'ADV', 'R', 'OLDVPRED', 'RWD', 'OLDNEGLOGPAC', 'WEIGHT']
        self.dtypes = [self.template[0].dtype, self.template[1].dtype] + \
                      [self.template[-1].dtype] * 4 + \
                      [self.template[-2].dtype] + \
                      [np.float32]
        self.shapes = [[batch_size] + list(self.template[0].shape)] + \
                      [[batch_size] + list(self.template[1].shape)] + \
                      [[batch_size] + list(self.template[-1].shape)] * 4 + \
                      [[batch_size] + list(self.template[-2].shape)] + \
                      [[batch_size]]
        if self.rem.rollout_len > 1:
            names += ['M', 'S']
            self.dtypes += [t.dtype for t in self.template[2:4]]
            self.shapes += [[batch_size] + list(self.template[2].shape)] + \
                           [[batch_size // self.rem.rollout_len] + \
                            list(self.template[3].shape)]
        def gfunc():
            while True:
                yield self._get_data(batch_size)
        assert len(names) == len(self.dtypes)
        assert len(names) == len(self.shapes)
        for i in range(len(names)):
            print("Prefetch data '%s': %s %s" % \
                    (names[i], str(self.shapes[i]), str(self.dtypes[i])))
        self.struct = namedtuple('InputData', names)
        self.generator_func = gfunc
        devices = get_visible_devices(device_type='GPU')
        for dev in devices:
          tfdata = TFData(gfunc, self.dtypes, self.shapes,
                          n_worker=FLAGS.n_data_worker, device=dev).batch_data
          self.input_data[dev] = self.struct(*tfdata)

    def _get_data(self, batch_size):
        """ This function is used by generator_func """
        nrollout = batch_size // self.rem.rollout_len
        data, weight = self.server.get_data(nrollout)
        # concatenate as a batch
        # [nrollout, rollout_len, shape] -> [nrollout * rollout_len, shape]
        obs = flatten(data[0])
        act = flatten(data[1])
        if self.rem.rollout_len > 1: # rnn
            mask = flatten(data[2])
            states = data[3].take(0, axis=1) # data[3][:,0]
        rwd = flatten(data[-4])
        nlp = flatten(data[-3])
        val = flatten(data[-2])
        qvl = flatten(data[-1])
        wgt = weight.repeat(self.rem.rollout_len)
        if False: # Check nan!
            stop = True
            if np.isnan(np.min(obs)):
                print("NaN in obs!")
            elif np.isnan(np.min(act)):
                print("NaN in act!")
            elif np.isnan(np.min(rwd)):
                print("NaN in rwd!")
            elif np.isnan(np.min(nlp)):
                print("NaN in nlp!")
            elif np.isnan(np.min(val)):
                print("NaN in val!")
            elif np.isnan(np.min(qvl)):
                print("NaN in qvl!")
            elif np.isnan(np.min(wgt)):
                print("NaN in wgt!")
            elif np.min(wgt) < 1e-8:
                print("wgt near zero!")
            else:
                stop = False
            if stop:
                sys.stdout.flush()
                assert False
        adv = qvl - val # Advantage
        #adv = (adv - adv.mean(axis=0)) / (adv.std(axis=0) + 1e-8)
        #adv = adv / (np.sqrt(np.mean(adv**2)) + 1e-8)
        #wgt = weight / (np.sqrt(np.mean(weight**2)) + 1e-8)

        # This should match InputData defined at the beginning.
        if self.rem.rollout_len > 1:
            return obs, act, adv, qvl, val, rwd, nlp, wgt, mask, states, 
        else:
            return obs, act, adv, qvl, val, rwd, nlp, wgt

