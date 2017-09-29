# Copyright (c) Chris Beckham 2017

from __future__ import print_function
import keras
from keras import backend as K
from keras.layers import Layer, Dense, Conv2D, activations
from keras.models import Model
from keras.layers.wrappers import Wrapper
from keras import initializers, regularizers, constraints
from theano import tensor as T


class LayerNorm(Layer):
    """
    TODO: comment.
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
        # What are the dimensions of the gamma/beta params?
        # (bs, p) --> n_out_idx = 1 (-1)
        # (bs, seq, p) --> n_out_idx = 2 (-1)
        # (bs, f, h, w) --> n_out_idx = 1
        self.n_out_idx = -1 if len(input_shape) in [2,3] else 1
        # What axes do we normalise over?
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
        self._input_shape = input_shape
        super(LayerNorm, self).build(input_shape)

    def call(self, x):
        this_mean = K.mean(x, axis=self.axes, keepdims=True)
        this_var = K.var(x, axis=self.axes, keepdims=True)
        ret = (x - this_mean) / (this_var + K.epsilon())
        #input_shape = K.int_shape(x)
        input_shape = self._input_shape     # HACK: recurrentshop workaround
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
    TODO: comment.
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
            # Keras kernel is of the form (k,k,in_dim,out_dim), but if we 
            # compute the norm it is in the shape (1,1,1,10). We want it to be
            # in the shape (1,10,1,1), i.e. the 2nd axis is the feature map 
            # axis, and the axes with 1's are 'broadcast' axes.
            norm_w = norm_w.swapaxes(-1,-2).swapaxes(-2,-3)
            self.norm_w = norm_w
            #self.g = self.g.dimshuffle('x',0,'x','x')
            n_out = self.layer.filters
        else:
            self.norm_w = K.sqrt(K.sum(K.square(W), axis=0, keepdims=True))
            #self.g = self.g.dimshuffle('x',0)
            n_out = self.layer.units
        # Define our own bias term here.
        # TODO: don't add a bias if the wrapped layer never had one in the first place
        self.bias = self.add_weight(shape=(n_out,), name='bias',
                                    initializer=self.layer.bias_initializer,
                                    constraint=self.layer.bias_constraint,
                                    regularizer=self.layer.bias_regularizer)
        self.name = self.layer.name + "_weightnorm"
        super(WeightNorm, self).build(input_shape)
        
    def compute_output_shape(self, input_shape):
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
    
    
class BilinearUpsample2DLayer(Layer):
    """
    2D bilinear upsampling layer

    Performs 2D upsampling (using bilinear interpolation) over the two trailing
    axes of a 4D input tensor.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    scale_factor : integer
        The scale factor in each dimension.

    use_1D_kernel : bool
        Upsample rows and columns separately using 1D kernels, otherwise
        use a 2D kernel.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    References
    -----
    .. [1] Augustus Odena, Vincent Dumoulin, Chris Olah (2016):
           Deconvolution and checkerboard artifacts. Distill.
           http://distill.pub/2016/deconv-checkerboard/
    """
    def __init__(self, scale_factor, use_1D_kernel=True, **kwargs):
        assert K.backend() == 'theano'
        super(BilinearUpsample2DLayer, self).__init__(**kwargs)
        self.scale_factor = scale_factor
        self.use_1D_kernel = use_1D_kernel
        if self.scale_factor < 1:
            raise ValueError('Scale factor must be >= 1, not {0}'.format(
                self.scale_factor))
        if isinstance(self.scale_factor, tuple):
            raise ValueError('Scale factor must be a scalar, not a tuple')
            
    def build(self, input_shape):
        self._input_shape = input_shape
        super(BilinearUpsample2DLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        h = input_shape[2]*self.scale_factor \
            if input_shape[2] != None else None
        w = input_shape[3]*self.scale_factor \
            if input_shape[3] != None else None
        return input_shape[0:2] + tuple([h, w])

    def call(self, input):
        return T.nnet.abstract_conv.bilinear_upsampling(
            input,
            self.scale_factor,
            batch_size=self._input_shape[0],
            num_input_channels=self._input_shape[1],
            use_1D_kernel=self.use_1D_kernel)


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
    print("training...")
    tmp.fit(x=xfake, y=yfake, epochs=100, verbose=0)
    print("weight magnitudes before train:",  wt_before)
    print("weight magnitudes after train:",
          [ np.sum(elem.get_value()**2) for elem in tmp.weights ])
    
    
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
    print("training...")
    tmp.fit(x=xfake, y=yfake, epochs=100, verbose=0)
    print("weight magnitudes before train:",  wt_before)
    print("weight magnitudes after train:",
          [ np.sum(elem.get_value()**2) for elem in tmp.weights ])
    
if __name__ == '__main__':
    test_weight_norm_mlp()
    test_weight_norm_cnn()


def test_bilinear_upsampling():
    from keras.layers import Input
    from keras.models import Model
    import numpy as np
    inp = Input((3,28,28))
    ups = BilinearUpsample2DLayer(scale_factor=2)(inp)
    md = Model(inp, ups)
    fake_x = np.random.normal(0,1,size=(5,3,28,28)).astype("float32")
    ups_fake_x = md.predict(fake_x)
    assert ups_fake_x.shape == (5,3,28*2,28*2)

if __name__ == '__main__':
    test_bilinear_upsampling()
