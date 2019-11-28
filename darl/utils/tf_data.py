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

import tensorflow as tf


class TFData(object):
    dtypes = None
    shapes = None
    dataset = None
    iterator = None
    batch_data = None

    def __init__(self, generator_func, dtypes, shapes, n_worker, device=None):
        """ generator_func should be an infinite loop, and 'yield' the data instead of 'return' them. """
        self.dtypes = tuple([tf.as_dtype(each) for each in dtypes])
        self.shapes = tuple(shapes)
        # Construct dataset
        self.dataset = tf.data.Dataset.range(n_worker).repeat().apply(
                tf.contrib.data.parallel_interleave(
                    lambda x: tf.data.Dataset.from_generator(generator_func, self.dtypes, self.shapes),
                    cycle_length=n_worker, sloppy=True))  # parallel generators
        self.dataset = self.dataset.prefetch(n_worker)
        if device is not None:
            self.dataset = self.dataset.apply(tf.contrib.data.prefetch_to_device(device))
        self.iterator = self.dataset.make_one_shot_iterator()
        self.batch_data = self.iterator.get_next()

