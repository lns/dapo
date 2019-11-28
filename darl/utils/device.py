#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import (absolute_import, division, print_function, unicode_literals)

try:
    from future.builtins import ascii, filter, hex, map, oct, zip
except ImportError:
    pass

import tensorflow as tf

def get_visible_devices(device_type=None):
  """ Get visible devices by device type.

  Visible devices are affected by setting CUDA_VISIBLE_DEVICES.

    :param  device_type   Possible values are None(all type), 'CPU', 'GPU', 'XLA_CPU', 'XLA_GPU', etc.
  """
  from tensorflow.python.client import device_lib
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if device_type is None or x.device_type==device_type]

def group_allreduce(tensor_list, group_key, merge_op, final_op):
  """ Group AllReduce Op with native TF utilities.

  See pastebin/allreduce.py
  """
  from tensorflow.python.ops.collective_ops import all_reduce
  group_size = len(get_visible_devices(device_type='GPU'))
  if group_size==1:
    return tensor_list
  else:
    ret = []
    for i,t in enumerate(tensor_list):
      if t is None: # TODO: Why we may encounter None here?
        continue
      ret.append(all_reduce(t, group_size=group_size, group_key=group_key,
                            instance_key=100*group_key+i, merge_op=merge_op, final_op=final_op))
    return ret

def sync_params(params):
  assign_ops = []
  for i,v in enumerate(params):
    if i != 0:
      for j,t in enumerate(v):
        assign_ops.append(t.assign(params[0][j]))
  return assign_ops

def params_square_dist(params):
  losses = [[]] * len(params)
  for i,v in enumerate(params):
    if i != 0:
      for j,t in enumerate(v):
        losses[i].append(tf.reduce_mean(tf.squared_difference(t, params[0][j])))
  for i in range(len(losses)):
    losses[i] = tf.reduce_mean(losses[i])
  return losses

# From https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py
def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    if True:
      # Keep in mind that the Variables are redundant because they are shared
      # across towers. So .. we will just return the first tower's pointer to
      # the Variable.
      v = grad_and_vars[0][1]
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)
    else:
      for _, v in grad_and_vars:
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
  return average_grads

