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
import time
import numpy as np
import os.path as osp
import tensorflow as tf
from darl.utils import logger
from darl.utils.device import get_visible_devices, sync_params, average_gradients, params_square_dist

from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_float("total_samples", 4e8, "Total number of training samples")
flags.DEFINE_integer("pub_interval", 100, "Frequency of publishing model.")
flags.DEFINE_integer("log_interval", 100, "Frequency of printing log.")
flags.DEFINE_integer("save_interval", 100000, "Frequency of saving model.")
flags.DEFINE_integer("starting_step", 1024, "Starting step.")
flags.DEFINE_string("load_path", '', "Path for loading model.")

class Learner(object):
  data_server = None
  devices = [] # device.name
  model = [] # model

  def __init__(self, data_server, model_ctor, **kwargs):
    self.data_server = data_server
    self.devices = get_visible_devices(device_type='GPU')
    self.model = []
    # The variable_scope.reuse seems redundent here, as variables are constructed in variable_scope('model')
    with tf.variable_scope(tf.get_variable_scope()):
      for i, dev in enumerate(self.devices):
        mdl = model_ctor(input_data=data_server.input_data[dev], device=dev,
                         build_actor_net=(i==0), **kwargs)
        tf.get_variable_scope().reuse_variables()
        self.model.append(mdl)
    # Combine gradients
    # See https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py
    self.tower_grads = [mdl.grads_v for mdl in self.model]
    avg_grads = average_gradients(self.tower_grads)
    self.params = []
    for mdl in self.model:
      self.params.append([g_and_v[1] for g_and_v in mdl.grads_v])
    # [DEBUG] param difference across towers
    self.diff = params_square_dist(self.params)
    # Setup Optimizer
    self.lr_var = tf.placeholder(tf.float32, [])
    self.lr = lambda f: (FLAGS.final_lr + f * (FLAGS.base_lr - FLAGS.final_lr))
    trainer = tf.train.AdamOptimizer(learning_rate=self.lr_var, epsilon=1e-5)
    self.train_op = trainer.apply_gradients(avg_grads)

  def run(self, sess):
    # Load params
    if FLAGS.load_path != '':
      for mdl in self.model:
        mdl.load_params(FLAGS.load_path)
    model0 = self.model[0]
    # start
    tfirststart = time.time()
    # init global variables
    tf.global_variables_initializer().run(session=sess)
    # sync params
    #assign_ops = sync_params(self.params)
    #sess.run(assign_ops) # not needed
    for mdl in self.model:
      mdl.sess = sess
    # publish model
    while self.data_server.server.rem.total_steps < FLAGS.starting_step:
      model0.mutex.acquire()
      self.data_server.pub(model0.serialize())
      model0.mutex.release()
      time.sleep(1)
    # make checkpoints dir and save model_names
    logger.configure(dir='out/', format_strs=['stdout', 'tensorboard', 'csv'])
    checkdir = osp.join(logger.get_dir(), 'checkpoints')
    os.makedirs(checkdir, exist_ok=True)
    nupdates = int(FLAGS.total_samples) // (FLAGS.batch_size * len(self.devices))
    tstart = time.time()
    for update in xrange(1, nupdates + 1):
      mblossvals = []
      # learning rate
      frac = 1.0 - (update - 1.0) / nupdates
      lrnow = self.lr(frac)
      feed_dict = {self.lr_var: lrnow}
      for mdl in self.model:
        for k,v in mdl.feeds(update, nupdates).items():
          feed_dict[k] = v
      loss_vars = [mdl.loss_vars for mdl in self.model]
      losses, _ = sess.run([loss_vars, self.train_op], feed_dict=feed_dict)
      mblossvals.extend(losses) #mblossvals.append(model.train(update, nupdates))

      if update % FLAGS.pub_interval == 0:
        self.data_server.pub(model0.serialize())
      if update % FLAGS.log_interval == 0 or update == 1:
        tnow = time.time()
        fps = int(len(self.devices) * FLAGS.batch_size * FLAGS.log_interval / (tnow - tstart))
        logger.logkv("n_episodes", self.data_server.server.rem.total_episodes)
        logger.logkv("n_updates", update)
        logger.logkv("sample_generated", self.data_server.server.rem.total_steps)
        logger.logkv("sample_consumed", update * FLAGS.batch_size * len(self.devices))
        logger.logkv("fps", fps)
        logger.logkv('time_elapsed', tnow - tfirststart)
        lossvals = np.mean(mblossvals, axis=0)
        for (lossval, lossname) in zip(lossvals, model0.loss_names):
          logger.logkv(lossname, lossval)
        diff = sess.run(self.diff, feed_dict=feed_dict)
        print("param MSE: "+str(diff))
        logger.dumpkvs()
        tstart = time.time()
      if FLAGS.save_interval and logger.get_dir() and \
          (update % FLAGS.save_interval == 0 or update == 1):
        model0.save_params(osp.join(checkdir, 'iter%.8i.params' % update))
        model0.save_frozen(osp.join(checkdir, 'iter%.8i.frozen' % update))

