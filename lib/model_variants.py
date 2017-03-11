from .model import assemble_model as assemble_cycled_model, _l2
from keras.models import Model
from keras.layers import (Input,
                          Convolution2D)
import copy

def assemble_model(stage3=False, **model_kwargs):
    if stage3:
        assert(model_kwargs['input_shape']==(1, 512, 512))
        input = Input(shape=(1, 512, 512))
        num_filters = model_kwargs['input_num_filters']
        first_conv = Convolution2D(num_filters, 3, 3,
                               init=model_kwargs['init'], border_mode='same',
                               W_regularizer=_l2(model_kwargs['weight_decay']),
                               name='first_conv_stage3')(input)
        model_input = model_kwargs['initblock'](nb_filter=num_filters,
                                     subsample=True, upsample=False,
                                     skip=True,
                                     dropout=model_kwargs['dropout'],
                                     batch_norm=model_kwargs['batch_norm'],
                                     weight_decay=model_kwargs['weight_decay'],
                                     num_residuals=1,
                                     bn_kwargs=model_kwargs['bn_kwargs'],
                                     init=model_kwargs['init'],
                                     name='stage3_initblock')(first_conv)
        stage3_model_kwargs = copy.copy(model_kwargs)
        stage3_model_kwargs['input_shape'] = (num_filters, 256, 256)
        stage3_model_kwargs['use_first_conv'] = False
        model = assemble_cycled_model(**stage3_model_kwargs)
        model.name = 'output_0'
        output = model(first_conv)
        model = Model(input=input, output=output)
    else:
        model = assemble_cycled_model(**model_kwargs)
    return model
