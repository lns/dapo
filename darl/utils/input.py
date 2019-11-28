import tensorflow as tf
from gym.spaces import Discrete, Box

def make_input(ob_space, input_data=None, batch_size=None, name='Ob'):
    '''
    Build observation input with encoding depending on the 
    observation space type
    Params:
    
    ob_space: observation space (should be one of gym.spaces)
    input_data: named tuple
    batch_size: batch size for input (default is None, so that resulting input placeholder can take tensors with any batch size)
    name: tensorflow variable name for input placeholder

    returns: tuple (input_placeholder, processed_input_tensor)
    '''

    input_x = None if input_data is None else input_data.X
    if isinstance(ob_space, Discrete):
        if input_x is None:
            input_x  = tf.placeholder(shape=(batch_size,), dtype=tf.int32, name=name)
        else:
            assert isinstance(input_x, tf.Tensor)
            if batch_size is None:
                assert len(input_x.shape) == 1
            else:
                assert input_x.shape == (batch_size,)
            assert input_x.dtype == tf.int32
        processed_x = tf.to_float(tf.one_hot(input_x, ob_space.n))
        return input_x, processed_x

    elif isinstance(ob_space, Box):
        if input_x is None:
            input_shape = (batch_size,) + ob_space.shape
            input_x = tf.placeholder(shape=input_shape, dtype=ob_space.dtype, name=name)
        else:
            assert isinstance(input_x, tf.Tensor)
            if batch_size is None:
                assert input_x.shape[1:] == ob_space.shape
            else:
                assert input_x.shape == (batch_size,) + ob_space.shape
            assert input_x.dtype == ob_space.dtype
        # scale to [0,1]
        # TODO: Why this will cause NaN?
        #processed_x = (tf.to_float(input_x) - ob_space.low) / (ob_space.high - ob_space.low + 1e-5)
        processed_x = tf.to_float(input_x)
        return input_x, processed_x

    else:
        print(ob_space)
        raise NotImplementedError

 
