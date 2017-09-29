import copy
from keras.layers import (Input,
                          Activation,
                          Permute,
                          BatchNormalization,
                          Lambda,
                          Dense,
                          Reshape,
                          Bidirectional,
                          TimeDistributed)
from keras.layers.merge import concatenate as merge_concatenate
from keras.layers.merge import multiply as merge_multiply
from keras.layers.merge import add as merge_add
from keras.models import Model
from keras import backend as K
from .model import assemble_model as assemble_cycled_model
from .model import (_l2,
                    _unique,
                    _softmax)
from .callbacks import Dice
from .loss import dice_loss
from .blocks import (Convolution,
                     bottleneck,
                     basic_block,
                     basic_block_mp,
                     residual_block,
                     get_nonlinearity)
from .normalization_layers import (LayerNorm,
                                   WeightNorm)


def assemble_cnn(input_shape, num_classes, num_init_blocks, num_main_blocks,
                 main_block_depth, input_num_filters, mainblock=None, 
                 initblock=None, dropout=0., normalization=BatchNormalization,
                 weight_decay=None, norm_kwargs=None, init='he_normal',
                 output_activation='linear', ndim=2):
    
    block_kwargs = {'skip': True,
                    'dropout': dropout,
                    'weight_decay': weight_decay,
                    'norm_kwargs': norm_kwargs,
                    'init': init,
                    'ndim': ndim}
    
    '''
    Returns the depth of a mainblock for a given pooling level.
    '''
    def get_repetitions(level):
        if hasattr(main_block_depth, '__len__'):
            return main_block_depth[level]
        return main_block_depth
        
    input = x = Input(shape=input_shape)
    
    # UP (initial subsampling blocks)
    for b in range(num_init_blocks):
        x = residual_block(initblock,
                           filters=input_num_filters,
                           repetitions=1,
                           subsample=True,
                           upsample=False,
                           normalization=normalization,
                           **block_kwargs)(x)
        print("Discriminator - INIT UP {}: {}".format(b, x._keras_shape))
    
    # UP (resnet blocks)
    for b in range(num_main_blocks):
        x = residual_block(mainblock,
                           filters=input_num_filters*(2**b),
                           repetitions=get_repetitions(b),
                           subsample=True,
                           upsample=False,
                           normalization=normalization,
                           **block_kwargs)(x)
        print("Discriminator - MAIN UP {}: {}".format(b, x._keras_shape))
        
    # Output
    def GlobalAveragePooling2D(input):
        return Lambda(function=lambda x: K.mean(x.flatten(3), axis=2),
                      output_shape=lambda s: s[:2])(input)
    avgpool = GlobalAveragePooling2D(x)
    output = Dense(num_classes, kernel_initializer=init, activation='linear',
                   kernel_regularizer=_l2(weight_decay))(avgpool)
    output = Activation(output_activation)(output)
    
    # Model
    model = Model(inputs=input, outputs=output)
    
    return model


def assemble_base_model(**model_kwargs):
    assert(model_kwargs['num_outputs']==2)
    
    input_shape = model_kwargs['input_shape']
    model_input = Input(shape=input_shape)
    
    # Assemble first model (liver)    
    model_liver_kwargs = copy.copy(model_kwargs)
    model_liver_kwargs['num_classes'] = None
    model_liver = assemble_cycled_model(**model_liver_kwargs)
    liver_output_pre = model_liver(model_input)
    
    # Assemble second model on top (lesion)
    model_lesion_kwargs = copy.copy(model_kwargs)
    model_lesion_kwargs['num_outputs'] = 1
    model_lesion_kwargs['num_classes'] = None
    model_lesion_kwargs['input_shape'] = (liver_output_pre._keras_shape[1]\
                                          +input_shape[-3],)+input_shape[1:]
    model_lesion = assemble_cycled_model(**model_lesion_kwargs)
    
    # Connect first model to second
    lesion_input = merge_concatenate([model_input, liver_output_pre], 
                                        axis=1)
    lesion_output_pre = model_lesion(lesion_input)
    
    # Base model
    model = Model(inputs=model_input,
                  outputs=[lesion_output_pre, liver_output_pre])
    
    return model


def assemble_model_two_levels(adversarial=False, num_residuals_bottom=None,
                              discriminator_kwargs=None, **model_kwargs):
    assert(model_kwargs['num_outputs']==2)
        
    if discriminator_kwargs is None:
        discriminator_kwargs = {}
        
    input_shape = model_kwargs['input_shape']
    model_input = Input(shape=input_shape, name='model_input')

    # Assemble first model (liver)    
    model_liver_kwargs = copy.copy(model_kwargs)
    model_liver_kwargs['num_classes'] = None
    if num_residuals_bottom is not None:
        model_liver_kwargs['num_residuals'] = num_residuals_bottom
    model_liver = assemble_cycled_model(**model_liver_kwargs)
    liver_output_pre = model_liver(model_input)
    
    # Assemble second model on top (lesion)
    model_lesion_kwargs = copy.copy(model_kwargs)
    model_lesion_kwargs['num_outputs'] = 1
    model_lesion_kwargs['input_shape'] = (liver_output_pre._keras_shape[1]\
                                          +input_shape[-3],)+input_shape[1:]
    model_lesion = assemble_cycled_model(**model_lesion_kwargs)
    
    # Connect first model to second
    lesion_input = merge_concatenate([model_input, liver_output_pre], 
                                     axis=1)
    
    # Create classifier for liver
    if model_kwargs['num_classes'] is not None:
        liver_output = Convolution(
            filters=1, kernel_size=1,
            ndim=model_kwargs['ndim'],
            activation='linear',
            kernel_regularizer=_l2(model_kwargs['weight_decay']),
            name='classifier_conv_1')
        liver_output = liver_output(liver_output_pre)
        if model_kwargs['ndim']==2:
            liver_output = Permute((2,3,1))(liver_output)
        else:
            liver_output = Permute((2,3,4,1))(liver_output)
        liver_output = Activation('sigmoid',
                                  name='sigmoid_1')(liver_output)
        if model_kwargs['ndim']==2:
            liver_output_layer = Permute((3,1,2))
        else:
            liver_output_layer = Permute((4,1,2,3))
        liver_output_layer.name = 'output_1'
        liver_output = liver_output_layer(liver_output)
    else:
        liver_output = Activation('linear',
                                  name='output_1')(liver_output_pre)
    
    # Lesion classifier output
    model_lesion.name = 'output_0'
    lesion_output = model_lesion(lesion_input)
    
    # Create discriminators
    if adversarial:
        def make_trainable(model, trainable=True):
            for l in model.layers:
                if isinstance(l, Model):
                    make_trainable(l, trainable)
                else:
                    l.trainable = trainable
        
        # Assemble discriminators.
        disc_0 = assemble_cnn(**discriminator_kwargs)
        disc_1 = assemble_cnn(**discriminator_kwargs)
        
        # Create discriminator outputs for real data.
        input_disc_0_seg = Input(input_shape, name='input_disc_0_seg')
        input_disc_1_seg = Input(input_shape, name='input_disc_1_seg')
        input_disc_0 = merge_concatenate([input_disc_0_seg, model_input],
                                         axis=1)
        input_disc_1 = merge_concatenate([input_disc_1_seg, model_input],
                                         axis=1)
        out_disc_0 = disc_0(input_disc_0)
        out_disc_1 = disc_1(input_disc_1)
        
        # Create untrainable segmentation generator output.
        model_gen = Model(inputs=model_input,
                            outputs=[lesion_output, liver_output])
        make_trainable(model_gen, False)
        outputs_gen = model_gen(model_input)
        
        # Create discriminator outputs for training the discriminators.
        input_disc_0 = merge_concatenate([outputs_gen[0], model_input],
                                         axis=1)
        input_disc_1 = merge_concatenate([outputs_gen[1], model_input],
                                         axis=1)
        out_adv_0_d = disc_0(input_disc_0)
        out_adv_1_d = disc_1(input_disc_1)
        
        # Make discriminators untrainable, generator trainable.
        make_trainable(model_gen, True)
        make_trainable(disc_0, False)
        make_trainable(disc_1, False)
        
        # Create discriminator outputs for training the generator.
        outputs_gen = model_gen(model_input)
        input_disc_0 = merge_concatenate([outputs_gen[0], model_input],
                                         axis=1)
        input_disc_1 = merge_concatenate([outputs_gen[1], model_input],
                                         axis=1)
        out_adv_0_g = disc_0(input_disc_0)
        out_adv_1_g = disc_1(input_disc_1)
        
        # Name the outputs.
        def name_layer(tensor, name):
            return Activation('linear', name=name)(tensor)
        out_adv_0_d = name_layer(out_adv_0_d, 'out_adv_0_d')
        out_adv_1_d = name_layer(out_adv_1_d, 'out_adv_1_d')
        out_adv_0_g = name_layer(out_adv_0_g, 'out_adv_0_g')
        out_adv_1_g = name_layer(out_adv_1_g, 'out_adv_1_g')
        out_disc_0 = name_layer(out_disc_0, 'out_disc_0')
        out_disc_1 = name_layer(out_disc_1, 'out_disc_1')
    
    # Create aggregate model
    if adversarial:
        model = Model( \
            inputs=[model_input,
                    input_disc_0_seg,
                    input_disc_1_seg],
            outputs=[lesion_output, liver_output,
                     out_adv_0_d, out_adv_1_d,
                     out_adv_0_g, out_adv_1_g,
                     out_disc_0, out_disc_1])
    else:
        model = Model(inputs=model_input,
                      outputs=[lesion_output, liver_output])
    
    return model
    
    
def assemble_model_multi_slice(ms_ndim_out=3, **model_kwargs):
    assert(model_kwargs['ndim']==3)
    
    # Assemble base model.
    input_shape = model_kwargs['input_shape']
    input_shape_2D = (input_shape[0],)+input_shape[2:]
    model_kwargs_2D = copy.copy(model_kwargs)
    model_kwargs_2D['ndim'] = 2
    model_kwargs_2D['input_shape'] = input_shape_2D
    base_model = assemble_base_model(**model_kwargs_2D)
    
    # Instantiate parallel models, sharing weights.
    # NOTE: batch norm statistics are shared!
    input_multi_slice = Input(input_shape)
    lesion_output_pre = []
    liver_output_pre = []
    z_axis = 2
    def select(i):
        return Lambda(lambda x: x[:,:,i,:,:], output_shape=input_shape_2D)
    def expand():
        output_shape = (model_kwargs['input_num_filters'],
                        1,)+input_shape_2D[1:]
        return Lambda(lambda x: K.expand_dims(x, axis=z_axis),
                      output_shape=output_shape)
    for i in range(3):
        out_0, out_1 = base_model(select(i)(input_multi_slice))
        lesion_output_pre.append(expand()(out_0))
        liver_output_pre.append(expand()(out_1))
    lesion_output_pre = merge_concatenate(lesion_output_pre, axis=z_axis)
    liver_output_pre = merge_concatenate(liver_output_pre, axis=z_axis)
    if ms_ndim_out==2:
        flat_shape = (model_kwargs['input_num_filters']*3,)\
                     +input_shape_2D[1:]
        lesion_output_pre = Reshape(flat_shape)(lesion_output_pre)
        liver_output_pre = Reshape(flat_shape)(liver_output_pre)
    
    # Add convolutions to combine information across slices.
    nonlinearity = model_kwargs['nonlinearity']
    lesion_output_pre = Convolution( \
        filters=model_kwargs['input_num_filters'],
        kernel_size=3,
        ndim=ms_ndim_out,
        padding='same',
        weight_norm=model_kwargs['weight_norm'],
        kernel_regularizer=_l2(model_kwargs['weight_decay']),
        name='conv_3D_0')(lesion_output_pre)
    lesion_output_pre = get_nonlinearity(nonlinearity)(lesion_output_pre)
    liver_output_pre = Convolution( \
        filters=model_kwargs['input_num_filters'],
        kernel_size=3,
        ndim=ms_ndim_out,
        padding='same',
        weight_norm=model_kwargs['weight_norm'],
        kernel_regularizer=_l2(model_kwargs['weight_decay']),
        name='conv_3D_1')(liver_output_pre)
    liver_output_pre = get_nonlinearity(nonlinearity)(liver_output_pre)
    
    # Create classifier for lesion.
    if model_kwargs['num_classes'] is not None:
        lesion_output = Convolution(filters=1, kernel_size=1,
                          ndim=ms_ndim_out,
                          activation='linear',
                          kernel_regularizer=_l2(model_kwargs['weight_decay']),
                          name='classifier_conv_0')
        lesion_output = lesion_output(lesion_output_pre)
        if ms_ndim_out==2:
            lesion_output = Permute((2,3,1))(lesion_output)
        else:
            lesion_output = Permute((2,3,4,1))(lesion_output)
        lesion_output = Activation('sigmoid', name='sigmoid_0')(lesion_output)
        if ms_ndim_out==2:
            lesion_output_layer = Permute((3,1,2))
        else:
            lesion_output_layer = Permute((4,1,2,3))
        lesion_output_layer.name = 'output_0'
        lesion_output = lesion_output_layer(lesion_output)
    else:
        lesion_output = Activation('linear',
                                   name='output_0')(lesion_output_pre)
            
    # Create classifier for liver.
    if model_kwargs['num_classes'] is not None:
        liver_output = Convolution(filters=1, kernel_size=1,
                          ndim=ms_ndim_out,
                          activation='linear',
                          kernel_regularizer=_l2(model_kwargs['weight_decay']),
                          name='classifier_conv_1')
        liver_output = liver_output(liver_output_pre)
        if ms_ndim_out==2:
            liver_output = Permute((2,3,1))(liver_output)
        else:
            liver_output = Permute((2,3,4,1))(liver_output)
        liver_output = Activation('sigmoid', name='sigmoid_1')(liver_output)
        if ms_ndim_out==2:
            liver_output_layer = Permute((3,1,2))
        else:
            liver_output_layer = Permute((4,1,2,3))
        liver_output_layer.name = 'output_1'
        liver_output = liver_output_layer(liver_output)
    else:
        liver_output = Activation('linear',
                                  name='output_1')(liver_output_pre)
    
    # Final model.
    model = Model(inputs=input_multi_slice,
                  outputs=[lesion_output, liver_output])
    return model


class Bidirectional_(Bidirectional):
    def __init__(self, layer, merge_mode='concat', weights=None,
                 custom_objects=None, **kwargs):
        super(Bidirectional, self).__init__(layer, **kwargs)
        if merge_mode not in ['sum', 'mul', 'ave', 'concat', None]:
            raise ValueError('Invalid merge mode. '
                             'Merge mode should be one of '
                             '{"sum", "mul", "ave", "concat", None}')
        self.forward_layer = copy.copy(layer)
        config = layer.get_config()
        config['go_backwards'] = not config['go_backwards']
        self.backward_layer = layer.__class__.from_config(
                                         config, custom_objects=custom_objects)
        self.forward_layer.name = 'forward_' + self.forward_layer.name
        self.backward_layer.name = 'backward_' + self.backward_layer.name
        self.merge_mode = merge_mode
        if weights:
            nw = len(weights)
            self.forward_layer.initial_weights = weights[:nw // 2]
            self.backward_layer.initial_weights = weights[nw // 2:]
        self.stateful = layer.stateful
        self.return_sequences = layer.return_sequences
        self.supports_masking = True
    
    
def assemble_model_recurrent(input_shape, num_filters, num_classes, 
                             normalization=LayerNorm, norm_kwargs=None,
                             weight_norm=False, num_outputs=1,
                             weight_decay=0.0005, init='he_normal'):
    from recurrentshop import RecurrentModel
    assert(num_outputs==1)
        
    if norm_kwargs is None:
        norm_kwargs = {}

    # Inputs
    model_input = Input(batch_shape=input_shape, name='model_input')
    input_t = Input(batch_shape=(input_shape[0],)+input_shape[2:])
    hidden_input_t = Input(batch_shape=(input_shape[0],
                                        num_filters)+input_shape[3:])
    
    # Common convolution kwargs.
    convolution_kwargs = {'filters': num_filters,
                          'kernel_size': 3,
                          'ndim': 2,
                          'padding': 'same',
                          'weight_norm': weight_norm,
                          'kernel_initializer': init}
    
    # GRU input.
    x_t = Convolution(**convolution_kwargs,
                      kernel_regularizer=_l2(weight_decay),
                      activation='relu',
                      name=_unique('conv_x'))(input_t)
    if normalization is not None:
        x_t = normalization(**norm_kwargs)(x_t)
    
    # GRU block.
    gate_replace_x = Convolution(**convolution_kwargs,
                                 kernel_regularizer=_l2(weight_decay),
                                 activation='sigmoid',
                                 name=_unique('conv_gate_replace'))(x_t)
    #if normalization is not None:
        #gate_replace_x = normalization(**norm_kwargs)(gate_replace_x)
    gate_replace_h = Convolution(**convolution_kwargs,
                                 kernel_regularizer=_l2(weight_decay),
                                 activation='sigmoid',
                             name=_unique('conv_gate_replace'))(hidden_input_t)
    #if normalization is not None:
        #gate_replace_h = normalization(**norm_kwargs)(gate_replace_h)
    gate_replace = merge_add([gate_replace_x, gate_replace_h])
        
    gate_read_x = Convolution(**convolution_kwargs,
                              kernel_regularizer=_l2(weight_decay),
                              activation='sigmoid',
                              name=_unique('conv_gate_read'))(x_t)
    #if normalization is not None:
        #gate_read_x = normalization(**norm_kwargs)(gate_read_x)
    gate_read_h = Convolution(**convolution_kwargs,
                              kernel_regularizer=_l2(weight_decay),
                              activation='sigmoid',
                              name=_unique('conv_gate_read'))(hidden_input_t)
    #if normalization is not None:
        #gate_read_h = normalization(**norm_kwargs)(gate_read_h)
    gate_read = merge_add([gate_read_x, gate_read_h])
        
    hidden_read_t = merge_multiply([gate_read, hidden_input_t])
    #if normalization is not None:
        #hidden_read_t = normalization(**norm_kwargs)(hidden_read_t)
        
    mix_t_pre = merge_concatenate([x_t, hidden_read_t], axis=1)
    mix_t = Convolution(**convolution_kwargs,
                        kernel_regularizer=_l2(weight_decay),
                        activation='tanh',
                        name=_unique('conv_mix'))(mix_t_pre)
    #if normalization is not None:
        #mix_t = normalization(**norm_kwargs)(mix_t)
        
    lambda_inputs = [mix_t, hidden_input_t, gate_replace]
    hidden_t = Lambda(function=lambda ins: ins[2]*ins[0] + (1-ins[2])*ins[1],
                      output_shape=lambda x:x[0])(lambda_inputs)
    
    # GRU output.
    out_t = Convolution(**convolution_kwargs,
                        kernel_regularizer=_l2(weight_decay),
                        activation='relu',
                        name=_unique('conv_out'))(hidden_t)
    class_convolution_kwargs = copy.copy(convolution_kwargs)
    class_convolution_kwargs['filters'] = num_classes
    out_t = Convolution(**class_convolution_kwargs,
                        kernel_regularizer=_l2(weight_decay),
                        activation='linear',
                        name=_unique('conv_out'))(hidden_t)
    #if normalization is not None:
        #out_t = normalization(**norm_kwargs)(out_t)
    
    # Classifier.
    out_t = Permute((2,3,1))(out_t)
    if num_classes==1:
        out_t = Activation('sigmoid')(out_t)
    else:
        out_t = Activation(_softmax)(out_t)
    out_t = Permute((3,1,2))(out_t)
    
    # Make it a recurrent block.
    #
    # NOTE: a bidirectional 'stateful' GRU has states passed between blocks
    # of the reverse path in non-temporal order. Only the forward pass is
    # stateful in sequential/temporal order.
    cobject = {LayerNorm.__name__: LayerNorm}
    output_layer = Bidirectional_(RecurrentModel(input=input_t,
                                               initial_states=[hidden_input_t],
                                               output=out_t,
                                               final_states=[hidden_t],
                                               stateful=True,
                                               return_sequences=True),
                                  merge_mode='sum',
                                  custom_objects=cobject)
    output_layer.name = 'output_0'
    model = Model(inputs=model_input, outputs=output_layer(model_input))
    return model
                 

def assemble_model(model_type='two_levels', num_residuals_bottom=None,
                   ms_ndim_out=3, discriminator_kwargs=None, **model_kwargs):
    # Model types: two_levels, adversarial, multi_slice, recurrent, simple
    # (`num_residuals_bottom` only used with two_levels)
    # (`ms_ndim_out` only used with multi_slice)
    # (`discriminator_kwargs` only used with adversarial)
    if model_type=='two_levels':
        '''
        Two FCNs: FCN_0 and FCN_1. Both have an output. FCN_1 takes as input 
        both the model input and the pre-classifier hidden representation of 
        FCN_0.
        '''
        model = assemble_model_two_levels(
            adversarial=False,
            num_residuals_bottom=num_residuals_bottom,
            **model_kwargs)
    if model_type=='adversarial':
        '''
        A two_levels model with a discriminator on each output, implementing
        an adversarial loss.
        '''
        model = assemble_model_two_levels(
            adversarial=True,
            num_residuals_bottom=num_residuals_bottom,
            discriminator_kwargs=discriminator_kwargs,
            **model_kwargs)
    if model_type=='multi_slice':
        '''
        A two_levels model that takes multiple slices as input and outputs one
        or or multiple slices as output. Via a convolution, combines 
        information across slices at the pre-classifier layer for each output.
        '''
        model = assemble_model_multi_slice(
            ms_ndim_out=ms_ndim_out,
            **model_kwargs)
    if model_type=='recurrent':
        '''
        A simple, stateful, 2D convolutional GRU.
        '''
        model = assemble_model_recurrent(**model_kwargs)
    if model_type=='simple':
        '''
        A single FCN.
        '''
        model = assemble_cycled_model(**model_kwargs)
        
    return model

    
def freeze_weights(model):
    def freeze(model):
        for l in model.layers:
            l.trainable = False
            if l.__class__.__name__ == 'Model':
                freeze(l)
    freeze(model)


def load_weights(model, path):
    from keras import backend as K
    import h5py
    f = h5py.File(path, mode='r')
    if 'layer_names' not in f.attrs and 'model_weights' in f:
        f = f['model_weights']
        
    flattened_layers = model.layers
    filtered_layers = []
    for layer in flattened_layers:
        weights = layer.weights
        if weights:
            filtered_layers.append(layer)
    flattened_layers = filtered_layers

    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
    filtered_layer_names = []
    for name in layer_names:
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        if len(weight_names):
            filtered_layer_names.append(name)
    layer_names = filtered_layer_names

    # We batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        if k >= len(flattened_layers):
            break
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        weight_values = [g[weight_name] for weight_name in weight_names]
        layer = flattened_layers[k]
        symbolic_weights = layer.weights
        if len(weight_values) != len(symbolic_weights):
            raise ValueError('Layer #' + str(k) +
                                ' (named "' + layer.name +
                                '" in the current model) was found to '
                                'correspond to layer ' + name +
                                ' in the save file. '
                                'However the new layer ' + layer.name +
                                ' expects ' + str(len(symbolic_weights)) +
                                ' weights, but the saved weights have ' +
                                str(len(weight_values)) +
                                ' elements.')
        weight_value_tuples += zip(symbolic_weights, weight_values)
    K.batch_set_value(weight_value_tuples)
    
    if hasattr(f, 'close'):
        f.close()
