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
from darl.model.policies import make_policy, FFNet, LSTMNet, PGHead
from darl.model.tf_model import TFModel
from darl.utils.distributions import make_pdtype
from darl.utils.utils import vtrace, truncIS, batch_to_seq, seq_to_batch
from darl.utils.device import group_allreduce
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_enum("adv_est", 'on', ['off','on'], "Use on-policy(v-trace)/off-policy(td-lambda) estimation of advantage")
flags.DEFINE_float("adv_coef", 1.0, "coefficient for advantage, the old BETA")
flags.DEFINE_float("adv_off", 0.0, "offset for advantage. Positive value encourage exploration.")
flags.DEFINE_float("reg_coef", 0.0, "coefficient for divergence regularization")
flags.DEFINE_float("ent_coef", 0.0, "coefficient for entropy regularization")
flags.DEFINE_float("cbarD", 1.0, "cbarD for V-trace like estimation of divergence")
flags.DEFINE_float("vf_coef", 0.5, "coefficient for entropy regularization")
flags.DEFINE_float("max_grad_norm", 0.5, "Max norm of gradient")
# rnn is duplicate with rollout_len > 1
flags.DEFINE_bool("rnn", True, "whether to use RNN")
flags.DEFINE_enum("value_loss", 'vanilla', ['vanilla', 'clipped', 'vtrace'], "type of value loss")
flags.DEFINE_enum("policy_loss", 'pgis', ['pg','pgis','acer','acerg','ppo','ppg','sil','marwil','vrmarwil'], "type of policy loss")
flags.DEFINE_float("acer_c", 1.0, "coefficient c in ACER (9)")
flags.DEFINE_enum("reg", 'KL', ['Entropy','KL','rKL','INF','TV','Hellinger'], "type of regularizer")
flags.DEFINE_float("gamma", 0.99, "Discount factor")
flags.DEFINE_integer("nlstm", 256, "Hidden dimension of lstm")
flags.DEFINE_float("base_lr", 1e-3, "base learing rate")
flags.DEFINE_float("final_lr", 0, "final learing rate")
flags.DEFINE_enum("ratio_loss", 'partial', ['kernel', 'full', 'partial', 'full0'], "type of ratio loss")
flags.DEFINE_integer("batch_size", 1024, "Number of samples in a batch")

def constfn(val):
  def f(_):
    return val
  return f

def as_func(obj):
  if isinstance(obj, float):
    return constfn(obj)
  else:
    assert callable(obj)
    return obj

class Model(TFModel):
  def __init__(self, base_net, ob_space, ac_space, device, build_actor_net, input_data):
    print('adv_est: %s'       % str(FLAGS.adv_est))
    print('adv_coef: %s'      % str(FLAGS.adv_coef))
    print('adv_off: %s'       % str(FLAGS.adv_off))
    print('reg_coef: %s'      % str(FLAGS.reg_coef))
    print('ent_coef: %s'      % str(FLAGS.ent_coef))
    print('cbarD: %s'         % str(FLAGS.cbarD))
    print('vf_coef: %s'       % str(FLAGS.vf_coef))
    print('max_grad_norm: %s' % str(FLAGS.max_grad_norm))
    print('rnn: %s'           % str(FLAGS.rnn))
    print('value_loss: %s'    % str(FLAGS.value_loss))
    print('gamma: %s'         % str(FLAGS.gamma))
    print('batch_size: %s'    % str(FLAGS.batch_size))
    print('nlstm: %s'         % str(FLAGS.nlstm))

    rollout_len = FLAGS.rollout_len
    reg_coef = FLAGS.reg_coef
    batch_size = FLAGS.batch_size

    if FLAGS.rnn:
      if build_actor_net:
        actor_net = LSTMNet(ob_space=ob_space, nbatch=1, base_net=base_net,
            input_data=None, reuse=False, rollout_len=1, nlstm=FLAGS.nlstm)
      with tf.device(device):
        train_net = LSTMNet(ob_space=ob_space, nbatch=batch_size, base_net=base_net,
            input_data=input_data, reuse=True, rollout_len=rollout_len, nlstm=FLAGS.nlstm)
    else:
      if build_actor_net:
        actor_net = FFNet(ob_space=ob_space, nbatch=1, base_net=base_net,
            input_data=None, reuse=False)
      with tf.device(device):
        train_net = FFNet(ob_space=ob_space, nbatch=batch_size, base_net=base_net,
            input_data=input_data, reuse=True)
    if build_actor_net:
      actor_head = PGHead(ac_space, n_v=1, reuse=False)
      actor_model = make_policy(actor_net, actor_head)
    with tf.device(device):
      train_head = PGHead(ac_space, n_v=1, reuse=True)
      train_model = make_policy(train_net, train_head)
    
      if FLAGS.rnn:
        X, A, ADV, R, OLDVPRED, RWD, OLDNEGLOGPAC, WEIGHT, M, S = input_data
      else:
        X, A, ADV, R, OLDVPRED, RWD, OLDNEGLOGPAC, WEIGHT = input_data
      W = tf.maximum(WEIGHT, 1e-2)

      # Placeholders
      #LR = tf.placeholder(tf.float32, [])
      CLIPRANGE = tf.placeholder(tf.float32, [])
      # Placeholders feeder
      #lr = lambda f: (FLAGS.final_lr + f * (FLAGS.base_lr - FLAGS.final_lr))
      #cliprange = lambda f: f * 0.2
      cliprange = 0.2
      #lr = as_func(lr)
      cliprange = as_func(cliprange)

      mean_return = tf.reduce_mean(R/W)
      neglogpac = train_model.head.pd.neglogp(A)
      entropy = tf.reduce_mean(train_model.head.pd.entropy()/W)
      ratio = tf.exp(tf.clip_by_value(OLDNEGLOGPAC - neglogpac, -10.0, 10.0))
      static_ratio = tf.minimum(tf.stop_gradient(ratio), 3.0)
      assert batch_size % rollout_len == 0
      nrollout = batch_size // rollout_len
      seq_ratio = batch_to_seq(ratio, nrollout, rollout_len, flat=True)
      if FLAGS.rnn:
        MS = train_net.ms # TODO: This is a hack!

      # Preprocess adv
      # TODO: These operations can be further moved to get_data() to speed up the training.
      if FLAGS.rnn:
        V,Q = vtrace(batch_to_seq(RWD, nrollout, rollout_len, flat=True),
               batch_to_seq(train_model.head.vf, nrollout, rollout_len, flat=True),
               seq_ratio, MS, gam=FLAGS.gamma, cbar=1.0, rhobar=1.0,
               input_R=batch_to_seq(R, nrollout, rollout_len, flat=True))
        V = tf.stop_gradient(seq_to_batch(V, flat=True))
        Q = tf.stop_gradient(seq_to_batch(Q, flat=True))
      else:
        V = (R - ADV) + static_ratio * ADV # V_t + rho * Adv
        Q = R

      vpred = train_model.head.vf

      # normalize ADV
      if FLAGS.adv_est == 'off':
        adv = ADV/(tf.sqrt(tf.maximum(tf.reduce_mean(tf.square(ADV)),1e-8)) + 1e-4)
        adv = tf.stop_gradient(FLAGS.adv_coef * (adv - FLAGS.adv_off))
      elif FLAGS.adv_est == 'on':
        adv = (Q-vpred)/(tf.sqrt(tf.maximum(tf.reduce_mean(tf.square(Q-vpred)),1e-8)) + 1e-4)
        adv = tf.stop_gradient(FLAGS.adv_coef * (adv - FLAGS.adv_off))
      else:
        raise NotImplementedError
      
      if FLAGS.value_loss == 'vanilla':
        vf_losses = tf.square(vpred - R)
        vf_loss = .5 * tf.reduce_mean(static_ratio * vf_losses/W)
      elif FLAGS.value_loss == 'clipped':
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.head.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(static_ratio * tf.maximum(vf_losses1, vf_losses2)/W)
      elif FLAGS.value_loss == 'vtrace':
        vf_losses = tf.square(vpred - V)
        vf_loss = .5 * tf.reduce_mean(static_ratio * vf_losses/W)
      else:
        raise RuntimeError("Unknown value_loss: '%s'" % FLAGS.value_loss)

      if FLAGS.policy_loss == 'pg':
        pg_losses = -adv * (-neglogpac)
      elif FLAGS.policy_loss == 'pgis':
        pg_losses = -adv * ratio
      elif FLAGS.policy_loss == 'ppo':
        pg_losses1 = -adv * ratio
        pg_losses2 = -adv * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_losses = tf.maximum(pg_losses1, pg_losses2)
      elif FLAGS.policy_loss == 'ppg':
        pg_losses = 0
      elif FLAGS.policy_loss == 'acer':
        pg_losses1 = -adv * tf.stop_gradient(tf.minimum(ratio, FLAGS.acer_c)) * (-neglogpac)
        pg_losses2 = -adv * tf.stop_gradient(tf.maximum(0.0, (ratio-FLAGS.acer_c)/ratio)) * (-neglogpac)
        pg_losses = pg_losses1 + pg_losses2
      elif FLAGS.policy_loss == 'acerg':
        pg_losses = 0 
      elif FLAGS.policy_loss == 'sil':
        pg_losses = tf.maximum(adv, 0.0) * neglogpac
      elif FLAGS.policy_loss == 'marwil':
        pg_losses = tf.exp(tf.clip_by_value(adv, -3.0, 3.0)) * neglogpac
      elif FLAGS.policy_loss == 'vrmarwil':
        pg_losses = tf.exp(tf.clip_by_value(adv, -3.0, 3.0)) * neglogpac + ratio
      else:
        raise RuntimeError("Unknown policy_loss: '%s'" % FLAGS.policy_loss)
        
      # \nabla_\theta D(\mu_\pi, \mu_t) = reg * \nabla_\theta \mu_\pi
      if FLAGS.reg == 'Entropy':
        reg = -neglogpac
      elif FLAGS.reg == 'KL':
        reg = OLDNEGLOGPAC - neglogpac
      elif FLAGS.reg == 'rKL': # reverse KL
        # We can add a constant 1 here, because ratio's gradient's expectation is zero
        reg = 1 - tf.exp(tf.clip_by_value(neglogpac-OLDNEGLOGPAC, -3.0, 3.0))
      elif FLAGS.reg == 'INF': # Implicitly Normalized Forecaster
        reg = tf.exp(tf.clip_by_value(0.5*OLDNEGLOGPAC, -3.0, 3.0)) - tf.exp(tf.clip_by_value(0.5*neglogpac, -3.0, 3.0))
      elif FLAGS.reg == 'Hellinger': # Hellinger distance. TODO: Check this
        reg = 1 - tf.exp(tf.clip_by_value(0.5*(neglogpac-OLDNEGLOGPAC), -3.0, 3.0))
      elif FLAGS.reg == 'TV': # Total Variance
        reg = 0.5 * tf.exp(tf.clip_by_value(-OLDNEGLOGPAC, -3.0, 3.0)) * tf.sign(ratio-1)
      else:
        raise RuntimeError("Unknown reg: '%s'" % FLAGS.reg)
      if FLAGS.rnn:
        reg = truncIS(batch_to_seq(reg, nrollout, rollout_len, flat=True),
                seq_ratio, MS, gam=FLAGS.gamma, cbar=FLAGS.cbarD, rhobar=1.0)
        reg = seq_to_batch(reg, flat=True)
      # TODO: This is a hack
      if FLAGS.policy_loss == 'ppg':
        augadv = adv - FLAGS.reg_coef * tf.stop_gradient(reg)
        reg_losses1 = -augadv * ratio
        reg_losses2 = -augadv * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        reg_losses = tf.maximum(reg_losses1, reg_losses2)
      elif FLAGS.policy_loss == 'acerg':
        augadv = adv - FLAGS.reg_coef * tf.stop_gradient(reg)
        reg_losses1 = -augadv * tf.stop_gradient(tf.minimum(ratio, FLAGS.acer_c)) * (-neglogpac)
        reg_losses2 = -augadv * tf.stop_gradient(tf.maximum(0.0, (ratio-FLAGS.acer_c)/ratio)) * (-neglogpac)
        reg_losses = reg_losses1 + reg_losses2
      else:
        reg_losses = FLAGS.reg_coef * ratio * tf.stop_gradient(reg)
      #
      pi_loss = tf.reduce_mean((pg_losses + reg_losses)/W)
      approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC)/W)
      #clipfrac = tf.reduce_mean(tf.to_float(tf.greater((ratio - 1.0)*tf.sign(ADV), CLIPRANGE)))
      loss = pi_loss - entropy * FLAGS.ent_coef + vf_loss * FLAGS.vf_coef
      with tf.variable_scope('model'):
        params = tf.trainable_variables()
      grads = tf.gradients(loss, params)
      if FLAGS.max_grad_norm is not None:
        grads, _grad_norm = tf.clip_by_global_norm(grads, FLAGS.max_grad_norm)
      #grads = group_allreduce(grads, group_key=0, merge_op='Add', final_op='Div')
      # Remove none due to unconnected graph
      self.grads_v = list(filter(lambda x: (x[0] is not None), zip(grads, params)))
      #self.grads_v = list(zip(grads, params))
      #trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
      #self.train_op = trainer.apply_gradients(self.grads_v)

    # Stats
    self.loss_vars  = [pi_loss, vf_loss, entropy, approxkl, mean_return]
    self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'mean_return']

    def feeds(update, nupdates):
      frac = 1.0 - (update - 1.0) / nupdates
      cliprangenow = cliprange(frac)
      td_map = {CLIPRANGE: cliprangenow}
      return td_map

    if build_actor_net:
      self.names = actor_model.names
    self.feeds = feeds

