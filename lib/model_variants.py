import copy
from keras.layers import (Input,
                          Activation,
                          Permute,
                          BatchNormalization,
                          Lambda,
                          Dense)
from keras.layers.merge import concatenate as merge_concatenate
from keras.models import Model
from keras import backend as K
from .model import assemble_model as assemble_cycled_model
from .model import _l2, _unique, _softmax
from .callbacks import Dice
from .loss import dice_loss
from .blocks import (Convolution,
                     bottleneck,
                     basic_block,
                     basic_block_mp,
                     residual_block,
                     get_nonlinearity)


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
    model_lesion_kwargs['input_shape'] = \
                                (liver_output_pre._keras_shape[1]+1,)\
                                +input_shape[1:]
    model_lesion = assemble_cycled_model(**model_lesion_kwargs)
    
    # Connect first model to second
    lesion_input = merge_concatenate([model_input, liver_output_pre], 
                                        axis=1)
    lesion_output_pre = model_lesion(lesion_input)
    
    # Base model
    model = Model(inputs=model_input,
                  outputs=[lesion_output_pre, liver_output_pre])
    
    return model
                 

def assemble_model(two_levels=False, num_residuals_bottom=None,
                   adversarial=False, multi_slice=False,
                   discriminator_kwargs=None, **model_kwargs):
    
    if not two_levels:
        model = assemble_cycled_model(**model_kwargs)
        return model
    
    # two_levels
    assert(model_kwargs['num_outputs']==2)
        
    if discriminator_kwargs is None:
        discriminator_kwargs = {}
        
    if not multi_slice:
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
        model_lesion_kwargs['input_shape'] = \
                                    (liver_output_pre._keras_shape[1]+1,)\
                                    +input_shape[1:]
        model_lesion = assemble_cycled_model(**model_lesion_kwargs)
        
        # Connect first model to second
        lesion_input = merge_concatenate([model_input, liver_output_pre], 
                                         axis=1)
        
        # Create classifier for liver
        liver_output = Convolution(filters=1, kernel_size=1,
                          ndim=model_kwargs['ndim'],
                          activation='linear',
                          kernel_regularizer=_l2(model_kwargs['weight_decay']),
                          name='classifier_conv_1')
        liver_output = liver_output(liver_output_pre)
        if model_kwargs['ndim']==2:
            liver_output = Permute((2,3,1))(liver_output)
        else:
            liver_output = Permute((2,3,4,1))(liver_output)
        liver_output = Activation('sigmoid', name='sigmoid_1')(liver_output)
        if model_kwargs['ndim']==2:
            liver_output_layer = Permute((3,1,2))
        else:
            liver_output_layer = Permute((4,1,2,3))
        liver_output_layer.name = 'output_1'
        liver_output = liver_output_layer(liver_output)
        
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
    
    if multi_slice:
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
        #inputs = []
        for i in range(3):
            #input_i = Input(input_shape_2D)
            #inputs.append(input_i)
            #out_0, out_1 = base_model(input_i)
            out_0, out_1 = base_model(select(i)(input_multi_slice))
            lesion_output_pre.append(expand()(out_0))
            liver_output_pre.append(expand()(out_1))
        lesion_output_pre = merge_concatenate(lesion_output_pre, axis=z_axis)
        liver_output_pre = merge_concatenate(liver_output_pre, axis=z_axis)
        print("DEBUG", lesion_output_pre._keras_shape,
              liver_output_pre._keras_shape)
        
        # Add 3D convolutions to combine information across slices.
        nonlinearity = model_kwargs['nonlinearity']
        lesion_output_pre = Convolution( \
            filters=model_kwargs['input_num_filters'],
            kernel_size=3,
            ndim=3,
            padding='same',
            weight_norm=model_kwargs['weight_norm'],
            kernel_regularizer=_l2(model_kwargs['weight_decay']),
            name='conv_3D_0')(lesion_output_pre)
        lesion_output_pre = get_nonlinearity(nonlinearity)(lesion_output_pre)
        liver_output_pre = Convolution( \
            filters=model_kwargs['input_num_filters'],
            kernel_size=3,
            ndim=3,
            padding='same',
            weight_norm=model_kwargs['weight_norm'],
            kernel_regularizer=_l2(model_kwargs['weight_decay']),
            name='conv_3D_1')(liver_output_pre)
        liver_output_pre = get_nonlinearity(nonlinearity)(liver_output_pre)
        
        # Create classifier for lesion.
        lesion_output = Convolution(filters=1, kernel_size=1,
                          ndim=3,
                          activation='linear',
                          kernel_regularizer=_l2(model_kwargs['weight_decay']),
                          name='classifier_conv_0')
        lesion_output = lesion_output(lesion_output_pre)
        lesion_output = Permute((2,3,4,1))(lesion_output)
        lesion_output = Activation('sigmoid', name='sigmoid_0')(lesion_output)
        lesion_output_layer = Permute((4,1,2,3))
        lesion_output_layer.name = 'output_0'
        lesion_output = lesion_output_layer(lesion_output)
                
        # Create classifier for liver.
        liver_output = Convolution(filters=1, kernel_size=1,
                          ndim=3,
                          activation='linear',
                          kernel_regularizer=_l2(model_kwargs['weight_decay']),
                          name='classifier_conv_1')
        liver_output = liver_output(liver_output_pre)
        liver_output = Permute((2,3,4,1))(liver_output)
        liver_output = Activation('sigmoid', name='sigmoid_1')(liver_output)
        liver_output_layer = Permute((4,1,2,3))
        liver_output_layer.name = 'output_1'
        liver_output = liver_output_layer(liver_output)
        
        # Final model.
        model = Model(inputs=input_multi_slice,
                      outputs=[lesion_output, liver_output])
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
