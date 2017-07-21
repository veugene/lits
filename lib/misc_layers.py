from theano import tensor as T
import keras.backend as K
from keras.layers import Layer

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
