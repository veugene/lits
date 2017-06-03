# Copyright (c) Chris Beckham 2017

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from theano import tensor as T

class LayerNorm(Layer):
    def __init__(self, axes=(2,3),
                 scale=True,
                 center=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        self.axes = axes
        self.scale = scale
        self.center = center
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        super(LayerNormLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = [1] + list(input_shape[1::])
        shape = tuple(shape)
        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
            # TODO: make generic so it works with TF
            self.gamma = T.addbroadcast(self.gamma, 0)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
            # TODO: make generic so it works with TF
            self.beta = T.addbroadcast(self.beta, 0)
            
        super(LayerNormLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        this_mean = K.mean(x, axis=self.axes, keepdims=True)
        this_var = K.var(x, axis=self.axes, keepdims=True)
        ret = (x - this_mean) / (this_var + K.epsilon())
        if self.scale:
            ret = ret*self.gamma
        if self.center:
            ret = ret+self.beta
        return ret

    def compute_output_shape(self, input_shape):
        return input_shape
