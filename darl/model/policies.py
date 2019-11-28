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

import numpy as np
import tensorflow as tf
from darl.utils.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from darl.utils.distributions import make_pdtype
from darl.utils.input import make_input


def nature_cnn(unscaled_images, images_format='NHWC', **conv_kwargs):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    if images_format == 'NHWC':
        pass
    elif images_format == 'NCHW':
        scaled_images = tf.transpose(scaled_images, perm=[0,2,3,1])
    else:
        raise RuntimeError("Unknown images format")
    # We require the input format to be NHWC
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))

def mlp_net(x, nh=[], activ=tf.nn.tanh):
    """
    MLP net
    """
    layers = [tf.reshape(x, [x.shape[0], -1])]
    for i in range(len(nh)):
        h = activ(fc(layers[-1], 'fc%d' % i, nh=nh[i], init_scale=np.sqrt(2)))
        layers.append(h)
    return layers[-1]

class FFNet(object):
    """
    Feed-forward net
    """
    def __init__(self, ob_space, nbatch, base_net, input_data=None, reuse=False):
        self.X, processed_x = make_input(ob_space, input_data, nbatch)
        with tf.variable_scope("model", reuse=reuse):
            self.h = base_net(self.X)

        self.pub_names = {'X':self.X.name}

class LSTMNet(object):
    def __init__(self, ob_space, nbatch, base_net, input_data=None,
                 reuse=False, rollout_len=32, nlstm=256):
        assert nbatch % rollout_len == 0
        nrollout = nbatch // rollout_len
        self.X, processed_x = make_input(ob_space, input_data, nbatch)
        if input_data is None:
            self.M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
            self.S = tf.placeholder(tf.float32, [nrollout, nlstm * 2])  # states
        else:
            self.M = tf.to_float(input_data.M)
            self.S = tf.to_float(input_data.S)
        with tf.variable_scope("model", reuse=reuse):
            h = base_net(self.X)
            xs = batch_to_seq(h, nrollout, rollout_len)
            self.ms = batch_to_seq(self.M, nrollout, rollout_len)
            h5, self.snew = lstm(xs, self.ms, self.S, 'lstm1', nh=nlstm)
            self.h = seq_to_batch(h5)

        self.pub_names = {'X':self.X.name, 'M':self.M.name, 'S':self.S.name, 'snew':self.snew.name}

class PGHead(object):
    def __init__(self, ac_space, n_v=1, reuse=False):  # pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        self.n_v = n_v
        self.reuse = reuse

    def add_head(self, h):
        with tf.variable_scope("model", reuse=self.reuse):
            self.vf = fc(h, 'v', self.n_v)[:, 0] # TODO: This is a hack, to be fixed
            self.pd, _ = self.pdtype.pdfromlatent(h, init_scale=0.01)
            # See implementation details in page 9 of 1904.08473
            self.wf = tf.log(1+tf.exp(fc(h, 'w', 1))) # shape (batch_size, 1)

        self.a0 = self.pd.sample()
        self.neglogp0 = self.pd.neglogp(self.a0)
        self.pub_names = {'a':self.a0.name, 'v':self.vf.name, 'p':self.neglogp0.name}

def merge(d1, d2):
    d = {}
    for k in d1.keys():
        d[k] = d1[k]
    for k in d2.keys():
        d[k] = d2[k]
    return d

class make_policy(object):
    def __init__(self, net, head):
        head.add_head(net.h)
        self.head = head
        self.net = net
        self.names = merge(self.net.pub_names, self.head.pub_names)
