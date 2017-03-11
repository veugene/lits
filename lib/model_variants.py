from keras.layers import (Input,
                          Activation,
                          Permute,
                          Convolution2D,
                          merge)
from keras.models import Model
import copy
from .model import assemble_model as assemble_cycled_model
from .model import _l2
from .callbacks import Dice
from .loss import dice_loss

def assemble_model(two_levels=False, num_residuals_bottom=None,
                   y_net=False, y_net_liver_path=None, y_net_lesion_path=None,
                   y_net_freeze_liver=True, y_net_freeze_lesion=True,
                   y_net_extra_input=False,
                   **model_kwargs):
    if two_levels:
        assert(not y_net)
        assert(model_kwargs['num_outputs']==2)
        
        input_shape = model_kwargs['input_shape']
        model_input = Input(shape=input_shape)
        
        # Assemble first model (liver)    
        model_liver_kwargs = copy.copy(model_kwargs)
        model_liver_kwargs['num_classes'] = None
        if num_residuals_bottom is not None:
            model_liver_kwargs['num_residuals'] = num_residuals_bottom
        model_liver = assemble_cycled_model(**model_liver_kwargs)
        
        # Assemble second model on top (lesion)
        model_lesion_kwargs = copy.copy(model_kwargs)
        model_lesion_kwargs['num_outputs'] = 1
        model_lesion_kwargs['input_shape'] = \
                                    (model_kwargs['input_num_filters']+1,)\
                                    +input_shape[1:]
        model_lesion = assemble_cycled_model(**model_lesion_kwargs)
        
        # Connect first model to second
        liver_output_pre = model_liver(model_input)
        lesion_input = merge([model_input, liver_output_pre],
                              mode='concat', concat_axis=1)
        
        # Create classifier for liver
        liver_output = Convolution2D(1,1,1, activation='linear',
                            W_regularizer=_l2(model_kwargs['weight_decay']),
                            name='classifier_conv_1')(liver_output_pre)
        liver_output = Permute((2,3,1))(liver_output)
        liver_output = Activation('sigmoid',name='sigmoid_1')(liver_output)
        liver_output_layer = Permute((3,1,2))
        liver_output_layer.name = 'output_1'
        liver_output = liver_output_layer(liver_output)
        
        # Create aggregate model
        model_lesion.name = 'output_0'
        lesion_output = model_lesion(lesion_input)
        model = Model(input=model_input, output=[lesion_output,
                                                 liver_output])
        return model
    
    elif y_net:
        assert(model_kwargs['num_outputs']==1)
        input_shape = model_kwargs['input_shape']
        model_input = Input(shape=input_shape)
        
        # Assemble liver and lesion models
        model_top_kwargs = copy.copy(model_kwargs)
        model_top_kwargs['num_classes'] = None
        model_liver = assemble_cycled_model(**model_top_kwargs)
        model_lesion = assemble_cycled_model(**model_top_kwargs)
        
        # Load liver and lesions model weights
        load_weights(model_liver, y_net_liver_path)
        load_weights(model_lesion, y_net_lesion_path)
        
        # Freeze liver and lesion models
        if y_net_freeze_liver:
            freeze_weights(model_liver)
        if y_net_freeze_lesion:
            freeze_weights(model_lesion)
        
        # Assemble the model on top
        model_top_kwargs = copy.copy(model_kwargs)
        model_top_kwargs['input_shape'] =  (model_liver.output_shape[1] \
                                           +model_lesion.output_shape[1],)\
                                           +input_shape[1:]
        model_top = assemble_cycled_model(**model_top_kwargs)
        
        # Feed liver and lesion models into top model
        top_input = merge([model_liver(model_input),
                           model_lesion(model_input)],
                          mode='concat',
                          concat_axis=1)
        output = model_top(top_input)
        
        # Create the aggregate model
        model_top.name = 'output_0'
        model = Model(input=model_input, output=output)
        return model
        
    else:
        return assemble_model(**model_kwargs)
    
    
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
