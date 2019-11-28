"""Distributions

based on https://github.com/openai/baselines/blob/master/baselines/common/distributions.py """
import tensorflow as tf
import numpy as np
from darl.utils.utils import fc
from tensorflow.python.ops import math_ops

class Pd(object):
    """
    A particular probability distribution
    """
    def flatparam(self):
        raise NotImplementedError
    def mode(self):
        raise NotImplementedError
    def neglogp(self, x):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError
    def kl(self, other):
        raise NotImplementedError
    def entropy(self):
        raise NotImplementedError
    def sample(self):
        raise NotImplementedError
    def logp(self, x):
        return - self.neglogp(x)

class PdType(object):
    """
    Parametrized family of probability distributions
    """
    def pdclass(self):
        raise NotImplementedError
    def pdfromflat(self, flat):
        return self.pdclass()(flat)
    def pdfromlatent(self, latent_vector):
        raise NotImplementedError
    def param_shape(self):
        raise NotImplementedError
    def sample_shape(self):
        raise NotImplementedError
    def sample_dtype(self):
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape+self.param_shape(), name=name)
    def sample_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape+self.sample_shape(), name=name)

    def __eq__(self, other):
        return (type(self) == type(other)) and (self.__dict__ == other.__dict__)


class CategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat
    def pdclass(self):
        return CategoricalPd
    def pdfromlatent(self, latent_vector, init_scale=1.0, init_bias=0.0, temperature=tf.to_float(1.0)):
        pdparam = fc(latent_vector, 'pi', self.ncat, init_scale=init_scale, init_bias=init_bias)
        pdparam = tf.scalar_mul(temperature, pdparam)
        return self.pdfromflat(pdparam), pdparam

    def param_shape(self):
        return [self.ncat]
    def sample_shape(self):
        return []
    def sample_dtype(self):
        return tf.int32


class MultiCategoricalPdType(PdType):
    def __init__(self, nvec):
        self.ncats = nvec
    def pdclass(self):
        return MultiCategoricalPd
    def pdfromflat(self, flat):
        return MultiCategoricalPd(self.ncats, flat)
    def param_shape(self):
        return [sum(self.ncats)]
    def sample_shape(self):
        return [len(self.ncats)]
    def sample_dtype(self):
        return tf.int32


class PixelCategoricalPdType(PdType):
    def __init__(self, w, h, dim):
        self.w = w
        self.h = h
        self.dim = dim
    def pdclass(self):
        return PixelCategoricalPd
    def pdfrompixel(self, pixel_logits):
        return PixelCategoricalPd(self.w, self.h, self.dim, pixel_logits)
    def sample_shape(self):
        return [self.w, self.h]
    def sample_dtype(self):
        return tf.int32


class DiagGaussianPdType(PdType):
    def __init__(self, size):
        self.size = size
    def pdclass(self):
        return DiagGaussianPd

    def pdfromlatent(self, latent_vector, init_scale=1.0, init_bias=0.0):
        mean = fc(latent_vector, 'pi', self.size, init_scale=init_scale, init_bias=init_bias)
        logstd = tf.get_variable(name='logstd', shape=[1, self.size], initializer=tf.zeros_initializer())
        pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        return self.pdfromflat(pdparam), mean

    def param_shape(self):
        return [2*self.size]
    def sample_shape(self):
        return [self.size]
    def sample_dtype(self):
        return tf.float32

class BernoulliPdType(PdType):
    def __init__(self, size):
        self.size = size
    def pdclass(self):
        return BernoulliPd
    def param_shape(self):
        return [self.size]
    def sample_shape(self):
        return [self.size]
    def sample_dtype(self):
        return tf.int32

class TuplePdType(PdType):
    def __init__(self, pdtype_list):
        self.pdtype_list = pdtype_list
        self.size = sum(self.param_shape())
        self.len = len(self.pdtype_list)

    def pdclass(self):
        return TuplePd

    def pdfromflat(self, flat):
        return self.pdclass()(flat, self.pdtype_list,
                              self.param_shape_list(),
                              self.sample_shape_list())

    def param_shape_list(self):
        return [pdtype.param_shape()[0] for pdtype in self.pdtype_list]

    def param_shape(self):
        return [sum(self.param_shape_list())]

    def sample_shape_list(self):
        return [1 if not pdtype.sample_shape() else pdtype.sample_shape()[0] for pdtype in self.pdtype_list]

    def sample_shape(self):
        return [sum(self.sample_shape_list())]

    def sample_dtype(self):
        return [pdtype.sample_dtype() for pdtype in self.pdtype_list]

# WRONG SECOND DERIVATIVES
# class CategoricalPd(Pd):
#     def __init__(self, logits):
#         self.logits = logits
#         self.ps = tf.nn.softmax(logits)
#     @classmethod
#     def fromflat(cls, flat):
#         return cls(flat)
#     def flatparam(self):
#         return self.logits
#     def mode(self):
#         return U.argmax(self.logits, axis=-1)
#     def logp(self, x):
#         return -tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, x)
#     def kl(self, other):
#         return tf.nn.softmax_cross_entropy_with_logits(other.logits, self.ps) \
#                 - tf.nn.softmax_cross_entropy_with_logits(self.logits, self.ps)
#     def entropy(self):
#         return tf.nn.softmax_cross_entropy_with_logits(self.logits, self.ps)
#     def sample(self):
#         u = tf.random_uniform(tf.shape(self.logits))
#         return U.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)

class CategoricalPd(Pd):
    def __init__(self, logits):
        self.logits = logits
    def flatparam(self):
        return self.logits
    def mode(self):
        return tf.argmax(self.logits, axis=-1)
    def neglogp(self, x):
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        one_hot_actions = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits,
            labels=one_hot_actions)
    def kl(self, other):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)
    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)
    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits))
        return tf.cast(tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1), tf.int32)
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class MultiCategoricalPd(Pd):
    def __init__(self, nvec, flat):
        self.flat = flat
        self.categoricals = list(map(CategoricalPd, tf.split(flat, nvec, axis=-1)))
    def flatparam(self):
        return self.flat
    def mode(self):
        return tf.cast(tf.stack([p.mode() for p in self.categoricals], axis=-1), tf.int32)
    def neglogp(self, x):
        return tf.add_n([p.neglogp(px) for p, px in zip(self.categoricals, tf.unstack(x, axis=-1))])
    def kl(self, other):
        return tf.add_n([p.kl(q) for p, q in zip(self.categoricals, other.categoricals)])
    def entropy(self):
        return tf.add_n([p.entropy() for p in self.categoricals])
    def sample(self):
        return tf.cast(tf.stack([p.sample() for p in self.categoricals], axis=-1), tf.int32)
    @classmethod
    def fromflat(cls, flat):
        raise NotImplementedError

class PixelCategoricalPd(Pd):
    def __init__(self, w, h, dim, pixel_logits):
        self.logits = pixel_logits  # [None, w, h, dim]
        self.w = w
        self.h = h
        self.dim = dim
        nvec = w * h
        self.vec_logits = tf.reshape(pixel_logits, [-1, nvec, dim])
    def neglogp(self, x):
        x = tf.reshape(x, [-1, self.w * self.h])
        neglogp = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.vec_logits, labels=x)
        return tf.reduce_sum(neglogp, axis=-1)
    def entropy(self):
        a0 = self.vec_logits - tf.reduce_max(self.vec_logits, axis=-1, keep_dims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keep_dims=True)
        p0 = ea0 / z0
        pixel_sum = tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)
        return tf.reduce_sum(pixel_sum, axis=-1)
    def sample(self):
        u = tf.random_uniform(tf.shape(self.vec_logits))
        a = tf.argmax(self.vec_logits - tf.log(-tf.log(u)), axis=-1)
        a = tf.reshape(a, [-1, self.w, self.h])
        return a
    @classmethod
    def fromflat(cls, flat):
        raise NotImplementedError

class DiagGaussianPd(Pd):
    def __init__(self, flat):
        self.flat = flat
        mean, logstd = tf.split(axis=len(flat.shape)-1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)
    def flatparam(self):
        return self.flat
    def mode(self):
        return self.mean
    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
               + tf.reduce_sum(self.logstd, axis=-1)
    def kl(self, other):
        assert isinstance(other, DiagGaussianPd)
        return tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) / (2.0 * tf.square(other.std)) - 0.5, axis=-1)
    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)
    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class BernoulliPd(Pd):
    def __init__(self, logits):
        self.logits = logits
        self.ps = tf.sigmoid(logits)
    def flatparam(self):
        return self.logits
    def mode(self):
        return tf.round(self.ps)
    def neglogp(self, x):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.to_float(x)), axis=-1)
    def kl(self, other):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=other.logits, labels=self.ps), axis=-1) - tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.ps), axis=-1)
    def entropy(self):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.ps), axis=-1)
    def sample(self):
        u = tf.random_uniform(tf.shape(self.ps))
        return tf.to_float(math_ops.less(u, self.ps))
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class TuplePd(Pd):
    def __init__(self, flat, pdtype_list, param_shape, sample_shape):
        self.flat = flat
        self.pdtype_list = pdtype_list
        self.param_shape = param_shape
        self.sample_shape = sample_shape
        self.flat_list = tf.split(self.flat, self.param_shape, axis=-1)
        self.pd_list = [pdtype.pdfromflat(flat) for pdtype, flat in zip(self.pdtype_list, self.flat_list)]
    def flatparam(self):
        return self.flat
    def neglogp(self, x):
        x_list = tf.split(x, self.sample_shape, axis=-1)
        return tf.stack([pd.neglogp(x) for pd, x in zip(self.pd_list, x_list)], axis=-1)
    def entropy(self):
        return tf.concat([pd.entropy() for pd in self.pd_list], axis=-1)
    def sample(self):
        samples = [pd.sample() for pd in self.pd_list]
        return tf.cast(tf.concat([sample if len(sample.shape)==2
                                  else tf.reshape(sample, (-1,1))
                                  for sample in samples], axis=-1), tf.int32)
    @classmethod
    def fromflat(cls, flat):
        raise NotImplementedError

def make_pdtype(ac_space):
    from gym import spaces
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        return DiagGaussianPdType(ac_space.shape[0])
    elif isinstance(ac_space, spaces.Discrete):
        return CategoricalPdType(ac_space.n)
    elif isinstance(ac_space, spaces.MultiDiscrete):
        return MultiCategoricalPdType(ac_space.nvec)
    elif isinstance(ac_space, spaces.MultiBinary):
        return BernoulliPdType(ac_space.n)
    elif isinstance(ac_space, spaces.Tuple):
        pd_list = []
        for space in ac_space.spaces:
            assert (isinstance(space, spaces.Discrete)
                    or isinstance(space, spaces.MultiDiscrete))
            pd_list.append(make_pdtype(space))
        return TuplePdType(pd_list)
    else:
        raise NotImplementedError

def shape_el(v, i):
    maybe = v.get_shape()[i]
    if maybe is not None:
        return maybe
    else:
        return tf.shape(v)[i]

