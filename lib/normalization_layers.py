# Copyright (c) Chris Beckham 2017

import keras
from keras import backend as K
from keras.layers import Layer, Dense, Conv2D, activations
from keras.models import Model
from keras.layers.wrappers import Wrapper
from keras import initializers, regularizers, constraints
from theano import tensor as T

class LayerNorm(Layer):
    """
    """
    def __init__(self,
                 scale=True,
                 center=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        self.scale = scale
        self.center = center
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

        super(LayerNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        # what are the dimensions of the gamma/beta params?
        # (bs, p) --> n_out_idx = 1 (-1)
        # (bs, seq, p) --> n_out_idx = 2 (-1)
        # (bs, f, h, w) --> n_out_idx = 1
        self.n_out_idx = -1 if len(input_shape) in [2,3] else 1
        # what axes do we normalise over?
        # (bs, p) --> axes = 1 (-1)
        # (bs, seq, p) --> axes = 2 (-1)
        # (bs, f, h, w) --> axes = (2,3)
        self.axes = -1 if len(input_shape) in [2,3] else (2,3)
        if self.scale:
            self.gamma = self.add_weight(shape=(input_shape[self.n_out_idx],),
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=(input_shape[self.n_out_idx],),
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        super(LayerNorm, self).build(input_shape)

    def call(self, x):
        this_mean = K.mean(x, axis=self.axes, keepdims=True)
        this_var = K.var(x, axis=self.axes, keepdims=True)
        ret = (x - this_mean) / (this_var + K.epsilon())
        input_shape = K.int_shape(x)
        dim_pattern = ['x'] * len(input_shape)
        dim_pattern[self.n_out_idx] = 0
        dim_pattern = tuple(dim_pattern)
        if self.scale:
            gamma = self.gamma.dimshuffle(*dim_pattern)
            ret = ret*gamma
        if self.center:
            beta = self.beta.dimshuffle(*dim_pattern)
            ret = ret+beta
        return ret

    def compute_output_shape(self, input_shape):
        return input_shape

class WeightNorm(Wrapper):
    """
    Compute weight norm as described in: ...
    """
    def __init__(self, layer, g_initializer='ones', **kwargs):
        if not ( isinstance(layer, Dense) or isinstance(layer, Conv2D) ):
            raise Exception("This wrapper currently only supports "
                           + "the wrapping of Dense and Conv2D")
        self.layer = layer
        super(WeightNorm, self).__init__(layer, **kwargs)

    def build(self, input_shape=None):
        if not self.layer.built:
            # remove the bias and nonlinearities
            # in case they are defined
            self.layer.use_bias = False
            self.activation = self.layer.activation
            self.layer.activation = activations.get('linear')
            self.layer.build(input_shape)
            self.layer.built = True
        W = self.layer.kernel
        self.g = self.add_weight(shape=(W.get_value().shape[-1]), name='g', initializer='ones')
        if len(input_shape) == 4:
            norm_w = K.sqrt(K.sum(K.square(W), axis=(0,1,2), keepdims=True))
            # keras kernel is of the form (k,k,in_dim,out_dim),
            # but if we compute the norm it is in the shape (1,1,1,10).
            # we want it to be in the shape (1,10,1,1), i.e. the 2nd axis
            # is the feature map axis, and the axes with 1's are 'broadcast' axes
            norm_w = norm_w.swapaxes(-1,-2).swapaxes(-2,-3)
            self.norm_w = norm_w
            #self.g = self.g.dimshuffle('x',0,'x','x')
            n_out = self.layer.filters
        else:
            self.norm_w = K.sqrt(K.sum(K.square(W), axis=0, keepdims=True))
            #self.g = self.g.dimshuffle('x',0)
            n_out = self.layer.units
        # define our own bias term here
        # TODO: don't add a bias if the wrapped layer never had one in the first place
        self.bias = self.add_weight(shape=(n_out,), name='bias',
                                    initializer=self.layer.bias_initializer,
                                    constraint=self.layer.bias_constraint,
                                    regularizer=self.layer.bias_regularizer)
        self.name = self.layer.name + "_weightnorm"
        super(WeightNorm, self).build(input_shape)
        
    def compute_output_shape(self, input_shape):
        #print dir(self.layer)
        return self.layer.compute_output_shape(input_shape)
    
    @property
    def trainable_weights(self):
        return self.layer.trainable_weights + [self.g, self.bias]
    
    def call(self, input, training=None):
        pre_act = self.layer(input)
        input_shape = K.int_shape(input)
        if len(input_shape) == 4:
            g_dimshuffle = self.g.dimshuffle('x',0,'x','x')
            bias_dimshuffle = self.bias.dimshuffle('x',0,'x','x')
        else:
            g_dimshuffle = self.g.dimshuffle('x',0)
            bias_dimshuffle = self.bias.dimshuffle('x',0)
        pre_act_norm = (pre_act / (K.epsilon() + self.norm_w)) * g_dimshuffle
        return self.activation(pre_act_norm + bias_dimshuffle)

def test_weight_norm_mlp():
    import numpy as np
    from keras.layers import Input
    xfake = np.random.normal(0,1, size=(100,28)).astype("float32")
    yfake = np.random.random(size=(100,5))
    yfake /= np.sum(yfake, axis=1, keepdims=True)
    inp = Input(shape=(28,))
    x = WeightNorm( Dense(64, activation='relu') )(inp)
    softmax = Dense(5, activation='softmax')(x)
    tmp = Model(inp, softmax)
    tmp.summary()
    wt_before = [ np.sum(elem.get_value()**2) for elem in tmp.weights ]
    tmp.compile(optimizer='adam', loss='categorical_crossentropy')
    print "training..."
    tmp.fit(x=xfake, y=yfake, epochs=100, verbose=0)
    print "weight magnitudes before train:",  wt_before
    print "weight magnitudes after train:",  [ np.sum(elem.get_value()**2) for elem in tmp.weights ]
    
def test_weight_norm_cnn():
    import numpy as np
    from keras.layers import Flatten, Input
    xfake = np.random.normal(0,1, size=(100,3,28,28)).astype("float32")
    yfake = np.random.random(size=(100,5))
    yfake /= np.sum(yfake, axis=1, keepdims=True)
    inp = Input(shape=(3,28,28))
    x = WeightNorm( Conv2D(filters=10, kernel_size=10, strides=2 ) )(inp)
    x = Flatten()(x)
    softmax = Dense(5, activation='softmax')(x)
    tmp = Model(inp, softmax)
    tmp.summary()
    wt_before = [ np.sum(elem.get_value()**2) for elem in tmp.weights ]
    tmp.compile(optimizer='adam', loss='categorical_crossentropy')
    print "training..."
    tmp.fit(x=xfake, y=yfake, epochs=100, verbose=0)
    print "weight magnitudes before train:",  wt_before
    print "weight magnitudes after train:",  [ np.sum(elem.get_value()**2) for elem in tmp.weights ]
    
if __name__ == '__main__':
    test_weight_norm_mlp()
    test_weight_norm_cnn()
