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
    import pickle
else:
    import cPickle as pickle

import tensorflow as tf
from threading import Lock
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("pickle_protocol", 2, "Pickle Protocol. Set to 2 to be compatiable with python2.3+")

def dump_graph(node_names, sess):
    """ Dump the graph related to node_names as graph_pb from session sess """
    # Actually, the function call for optimizing a model for inference in TF
    # is actively evolving, possible options including TransfromGraph and
    # optimize_for_inference_lib. A few related discussions can be found at
    # [1] https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md
    # [2] https://medium.com/@prasadpal107/saving-freezing-optimizing-for-inference-restoring-of-tensorflow-models-b4146deb21b5
    # [3] https://medium.com/google-cloud/optimizing-tensorflow-models-for-serving-959080e9ddbf
    graph_def = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph.as_graph_def(),
            [name.split(':',1)[0] for name in node_names])
    return graph_def.SerializeToString() # graph_pb

def load_graph(graph_pb, session_target='', session_config=None):
    """ Load a graph_pb into a new session """
    with tf.Graph().as_default() as graph:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(graph_pb)
        tf.import_graph_def(graph_def, name='prefix')
    sess = tf.Session(graph=graph, target=session_target, config=session_config)
    return sess

class TFModel(object):
    ''' TFModel is not thread-safe. Please use the mutex when using it. '''
    # Put the input and output node names in self.names, as a dict.
    names = None  # key -> tensor.name
    nodes = None  # key -> tensor
    sess  = None  # tf.Session()
    mutex = Lock()
    _new_params = None
    _assign_ops = None
    session_target = ''
    session_config = None

    def __init__(self, session_target='', session_config=None):
        self.session_target = session_target
        self.session_config = session_config

    def serialize(self):
        """ Serialize the graph in self.sess for inference. """
        assert self.names is not None
        assert self.sess is not None
        graph_pb = dump_graph(self.names.values(), self.sess)
        serial = pickle.dumps((self.names, graph_pb), protocol=FLAGS.pickle_protocol)
        return serial

    def deserialize(self, serial):
        """ Deserialize serial to a new session as self.sess. """
        self.names, graph_pb = pickle.loads(serial)
        if self.sess is not None:
            self.sess.close()
        self.sess = load_graph(graph_pb,
                session_target=self.session_target, session_config=self.session_config)
        # Assign self.nodes
        self.nodes = {}
        for k in self.names:
            self.nodes[k] = self.sess.graph.get_tensor_by_name('prefix/'+self.names[k])
        #sess.__enter__() # Is this a hack?
        # See https://stackoverflow.com/questions/49564185/tensorflow-setting-default-session-using-as-default-enter

    def save_params(self, file_path, scope=None):
        """ Save trainable params to a file.
        This method should only be called after constructing the whole graph. """
        params = tf.trainable_variables(scope=scope)
        weights = self.sess.run(params)
        with open(file_path, 'wb') as f:
            pickle.dump(weights, f, protocol=FLAGS.pickle_protocol)

    def load_params(self, file_path):
        """ Load trainable params from a file.
        This method should only be called after constructing the whole graph. """
        if self._new_params is None:
            params = tf.trainable_variables(scope=scope)
            self._new_params = [tf.placeholder(p.dtype, shape=p.get_shape()) for p in params]
            self._assign_ops = [p.assign(np) for p, np in zip(params, self._new_params)] 
        weights = pickle.load(file_path)
        self.sess.run(self._assign_ops, feed_dict={p: v for p, v in zip(self._new_params, weights)})

    def save_frozen(self, file_path):
        """ Save the model as frozen. """
        with tf.gfile.GFile(file_path, 'wb') as f:
            f.write(self.serialize())

    def load_frozen(self, file_path):
        """ Load the model as frozen. """
        with tf.gfile.GFile(file_path, 'rb') as f:
            serial = f.read(n=-1)
        self.deserialize(serial)

