from keras.models import Model
from keras.layers import (Input,
                          Activation,
                          Dense,
                          Permute,
                          Lambda,
                          BatchNormalization)
from keras.layers.merge import add as merge_add
from .blocks import Convolution
from keras import backend as K
from theano import tensor as T
from keras.regularizers import l2
import numpy as np
from .blocks import (bottleneck,
                     basic_block,
                     basic_block_mp,
                     residual_block)


def _l2(decay):
    """
    Return a new instance of l2 regularizer, or return None
    """
    if decay is not None:
        return l2(decay)
    else:
        return None
    

def _softmax(x):
    """
    Softmax that works on ND inputs.
    """
    e = K.exp(x - K.max(x, axis=-1, keepdims=True))
    s = K.sum(e, axis=-1, keepdims=True)
    return e / s


def _unique(name):
    """
    Return a unique name string.
    """
    return name + '_' + str(K.get_uid(name))
    
    
def assemble_model(input_shape, num_classes, num_init_blocks, num_main_blocks,
                   main_block_depth, input_num_filters, num_cycles=1,
                   preprocessor_network=None, postprocessor_network=None,
                   mainblock=None, initblock=None, dropout=0.,
                   normalization=BatchNormalization, weight_decay=None,
                   norm_kwargs=None, init='he_normal', ndim=2,
                   cycles_share_weights=True, num_residuals=1,
                   num_first_conv=1, num_final_conv=1, num_classifier=1,
                   num_outputs=1, use_first_conv=True, use_final_conv=True):
    """
    input_shape : tuple specifiying the 2D image input shape.
    num_classes : number of classes in the segmentation output.
    num_init_blocks : the number of blocks of type initblock, above mainblocks.
        These blocks always have the same number of channels as the first
        convolutional layer in the model.
    num_main_blocks : the number of blocks of type mainblock, below initblocks.
        These blocks double (halve) in number of channels at each downsampling
        (upsampling).
    main_block_depth : an integer or list of integers specifying the number of
        repetitions of each mainblock. A list must contain as many values as
        there are main_blocks in the downward (or upward -- it's mirrored) path
        plus one for the across path.
    input_num_filters : the number channels in the first (last) convolutional
        layer in the model (and of each initblock).
    num_cycles : number of times to cycle the down/up processing pair.
    preprocessor_network : a neural network for preprocessing the input data.
    postprocessor_network : a neural network for postprocessing the data fed
        to the classifier.
    mainblock : a layer defining the mainblock (bottleneck by default).
    initblock : a layer defining the initblock (basic_block_mp by default).
    dropout : the dropout probability, introduced in every block.
    normalization : The normalization to apply to layers (by default: batch
        normalization). If None, no normalization is applied.
    weight_decay : the weight decay (L2 penalty) used in every convolution.
    norm_kwargs : keyword arguments to pass to batch norm layers.
    init : string or function specifying the initializer for layers.
    ndim : the spatial dimensionality of the input and output (2 or 3)
    cycles_share_weights : share network weights across cycles.
    num_residuals : the number of parallel residual functions per block.
    num_first_conv : the number of parallel first convolutions.
    num_final_conv : the number of parallel final convolutions (+BN).
    num_classifier : the number of parallel linear classifiers.
    num_outputs : the number of model outputs, each with num_classifier
        classifiers.
    """
    
    '''
    By default, use depth 2 basic_block for mainblock
    '''
    if mainblock is None:
        mainblock = basic_block
    if initblock is None:
        initblock = basic_block_mp
    
    '''
    main_block_depth can be a list per block or a single value 
    -- ensure the list length is correct (if list) and that no length is 0
    '''
    if not hasattr(main_block_depth, '__len__'):
        if main_block_depth==0:
            raise ValueError("main_block_depth must never be zero")
    else:
        if len(main_block_depth)!=num_main_blocks+1:
            raise ValueError("main_block_depth must have " 
                             "`num_main_blocks+1` values when " 
                             "passed as a list")
        for d in main_block_depth:
            if d==0:
                raise ValueError("main_block_depth must never be zero")
    
    '''
    Returns the depth of a mainblock for a given pooling level.
    '''
    def get_repetitions(level):
        if hasattr(main_block_depth, '__len__'):
            return main_block_depth[level]
        return main_block_depth
    
    '''
    Merge tensors, changing the number of feature maps in the first input
    to match that of the second input. Feature maps in the first input are
    reweighted.
    
    If weight sharing is enabled, reuse old convolutions.
    '''
    def merge_into(x, into, skips, cycle, direction, depth):
        if x._keras_shape[1] != into._keras_shape[1]:
            if cycles_share_weights and depth in skips[cycle-1][direction]:
                conv_layer = skips[cycle-1][direction][depth]
            else:
                name = _unique('long_skip_'+str(direction)+'_'+str(depth))
                conv_layer = Convolution(filters=into._keras_shape[1],
                                         kernel_size=1,
                                         ndim=ndim,
                                         kernel_initializer=init,
                                         padding='valid',
                                         kernel_regularizer=_l2(weight_decay),
                                         name=name)
            skips[cycle][direction][depth] = conv_layer
            x = conv_layer(x)
        
        out = merge_add([x, into])
        return out
    
    '''
    Given some block function and an input tensor, return a reusable model
    instantiating that block function. This is to allow weight sharing.
    '''
    def make_block(block_func, x):
        x_filters = x._keras_shape[1]
        input = Input(shape=(x_filters,)+tuple([None]*ndim))
        model = Model(input, block_func(input))
        return model
    
    '''
    Constant kwargs passed to the init and main blocks.
    '''
    block_kwargs = {'skip': True,
                    'dropout': dropout,
                    'weight_decay': weight_decay,
                    'num_residuals': num_residuals,
                    'norm_kwargs': norm_kwargs,
                    'init': init,
                    'ndim': ndim}
    if norm_kwargs is None:
        norm_kwargs = {}
    
    # INPUT
    input = Input(shape=input_shape)
    
    # Preprocessing
    if preprocessor_network is not None:
        input = preprocessor_network(input)
    
    '''
    Build the blocks for all cycles, contracting and expanding in each cycle.
    '''
    tensors = [] # feature tensors
    blocks = []  # residual block layers
    skips = []   # 1x1 kernel convolution layers on long skip connections
    x = input
    for cycle in range(num_cycles):
        # Create tensors and layer lists for this cycle.
        tensors.append({'down': {}, 'up': {}, 'across': {}})
        blocks.append({'down': {}, 'up': {}, 'across': {}})
        skips.append({'down': {}, 'up': {}, 'across': {}})
        
        # First convolution
        if cycle > 0:
            x = merge_into(x, tensors[cycle-1]['up'][0], skips=skips,
                           cycle=cycle, direction='down', depth=0)
        if cycles_share_weights and cycle > 1:
            block = blocks[cycle-1]['down'][0]
        else:
            def first_block(x):
                outputs = []
                for i in range(num_first_conv):
                    out = Convolution(filters=input_num_filters,
                                      kernel_size=3, ndim=ndim,
                                      kernel_initializer=init, padding='same',
                                      kernel_regularizer=_l2(weight_decay),
                                      name=_unique('first_conv_'+str(i)))(x)
                    outputs.append(out)
                if len(outputs)>1:
                    out = merge_add(outputs)
                else:
                    out = outputs[0]
                return out
            block = make_block(first_block, x)
        if use_first_conv:
            x = block(x)
            blocks[cycle]['down'][0] = block
        else:
            blocks[cycle]['down'][0] = lambda x:x
        tensors[cycle]['down'][0] = x
        print("Cycle {} - FIRST DOWN: {}".format(cycle, x._keras_shape))
        
        # DOWN (initial subsampling blocks)
        for b in range(num_init_blocks):
            depth = b+1
            if cycle > 0:
                x = merge_into(x, tensors[cycle-1]['up'][depth], skips=skips,
                               cycle=cycle, direction='down', depth=depth)
            if cycles_share_weights and cycle > 1:
                block = blocks[cycle-1]['down'][depth]
            else:
                block_func = residual_block(initblock,
                                            filters=input_num_filters,
                                            repetitions=1,
                                            subsample=True,
                                            upsample=False,
                                            normalization=normalization,
                                            name='d'+str(depth),
                                            **block_kwargs)
                block = make_block(block_func, x)
            x = block(x)
            blocks[cycle]['down'][depth] = block
            tensors[cycle]['down'][depth] = x
            print("Cycle {} - INIT DOWN {}: {}".format(cycle, b,
                                                       x._keras_shape))
        
        # DOWN (resnet blocks)
        for b in range(num_main_blocks):
            depth = b+1+num_init_blocks
            if cycle > 0:
                x = merge_into(x, tensors[cycle-1]['up'][depth], skips=skips,
                               cycle=cycle, direction='down', depth=depth)
            if cycles_share_weights and cycle > 1:
                block = blocks[cycle-1]['down'][depth]
            else:
                block_func = residual_block(mainblock,
                                            filters=input_num_filters*(2**b),
                                            repetitions=get_repetitions(b),
                                            subsample=True,
                                            upsample=False,
                                            normalization=normalization,
                                            name='d'+str(depth),
                                            **block_kwargs)
                block = make_block(block_func, x)
            x = block(x)
            blocks[cycle]['down'][depth] = block
            tensors[cycle]['down'][depth] = x
            print("Cycle {} - MAIN DOWN {}: {}".format(cycle, b,
                                                       x._keras_shape))
            
        # ACROSS
        if cycle > 0:
            x = merge_into(x, tensors[cycle-1]['across'][0], skips=skips,
                           cycle=cycle, direction='across', depth=0)
        if cycles_share_weights and cycle > 1:
            block = blocks[cycle-1]['across'][0]
        else:
            block_func = residual_block( \
                                  mainblock,
                                  filters=input_num_filters*(2**b),
                                  repetitions=get_repetitions(num_main_blocks),
                                  subsample=True,
                                  upsample=True,
                                  normalization=normalization,
                                  name='a',
                                  **block_kwargs)
            block = make_block(block_func, x)
        x = block(x)
        blocks[cycle]['across'][0] = block
        tensors[cycle]['across'][0] = x
        print("Cycle {} - ACROSS: {}".format(cycle, x._keras_shape))

        # UP (resnet blocks)
        for b in range(num_main_blocks-1, -1, -1):
            depth = b+1+num_init_blocks
            x = merge_into(x, tensors[cycle]['down'][depth], skips=skips,
                           cycle=cycle, direction='up', depth=depth)
            if cycles_share_weights and cycle > 0 and cycle < num_cycles-1:
                block = blocks[cycle-1]['up'][depth]
            else:
                
                block_func = residual_block(mainblock,
                                            filters=input_num_filters*(2**b),
                                            repetitions=get_repetitions(b),
                                            subsample=False,
                                            upsample=True,
                                            normalization=normalization,
                                            name='u'+str(depth),
                                            **block_kwargs)
                block = make_block(block_func, x)
            x = block(x)
            blocks[cycle]['up'][depth] = block
            tensors[cycle]['up'][depth] = x
            print("Cycle {} - MAIN UP {}: {}".format(cycle, b,
                                                     x._keras_shape))
        
        # UP (final upsampling blocks)
        for b in range(num_init_blocks-1, -1, -1):
            depth = b+1
            x = merge_into(x, tensors[cycle]['down'][depth], skips=skips,
                           cycle=cycle, direction='up', depth=depth)
            if cycles_share_weights and cycle > 0 and cycle < num_cycles-1:
                block = blocks[cycle-1]['up'][depth]
            else:
                block_func = residual_block(initblock,
                                            filters=input_num_filters,
                                            repetitions=1,
                                            subsample=False,
                                            upsample=True,
                                            normalization=normalization,
                                            name='u'+str(depth),
                                            **block_kwargs)
                block = make_block(block_func, x)
            x = block(x)
            blocks[cycle]['up'][depth] = block
            tensors[cycle]['up'][depth] = x
            print("Cycle {} - INIT UP {}: {}".format(cycle, b,
                                                     x._keras_shape))
            
        # Final convolution.
        x = merge_into(x, tensors[cycle]['down'][0], skips=skips,
                       cycle=cycle, direction='up', depth=0)
        if cycles_share_weights and cycle > 0 and cycle < num_cycles-1:
            block = blocks[cycle-1]['up'][0]
        else:
            def final_block(x):
                outputs = []
                for i in range(num_final_conv):
                    out = Convolution(filters=input_num_filters,
                                      kernel_size=3, ndim=ndim,
                                      kernel_initializer=init, padding='same',
                                      kernel_regularizer=_l2(weight_decay),
                                      
                                        name=_unique('final_conv_'+str(i)))(x)
                    if normalization is not None:
                        out = normalization(name=_unique('final_norm_'+str(i)),
                                            **norm_kwargs)(out)
                    out = Activation('relu')(out)
                    outputs.append(out)
                if len(outputs)>1:
                    out = merge_add(outputs)
                else:
                    out = outputs[0]
                return out
            block = make_block(final_block, x)
        if use_final_conv:
            x = block(x)
            blocks[cycle]['up'][0] = block
        else:
            blocks[cycle]['up'][0] = lambda x:x
        tensors[cycle]['up'][0] = x
        if cycle > 0:
            # Merge preclassifier outputs across all cycles.
            x = merge_into(x, tensors[cycle-1]['up'][0], skips=skips,
                           cycle=cycle, direction='up', depth=-1)
        print("Cycle {} - FIRST UP: {}".format(cycle, x._keras_shape))
            
    # Postprocessing
    if postprocessor_network is not None:
        x = postprocessor_network(x)
    
    # OUTPUTs (SIGMOID)
    all_outputs = []
    if num_classes is not None:
        for i in range(num_outputs):
            # Linear classifier
            classifiers = []
            for j in range(num_classifier):
                name = 'classifier_conv_'+str(j)
                if i > 0:
                    # backwards compatibility
                    name += '_out'+str(i)
                output = Convolution(filters=num_classes, kernel_size=1, 
                                     ndim=ndim, activation='linear', 
                                     kernel_regularizer=_l2(weight_decay),
                                     name=_unique(name))(x)
                classifiers.append(output)
            if len(classifiers)>1:
                output = merge_add(classifiers)
            else:
                output = classifiers[0]
            if ndim==2:
                output = Permute((2,3,1))(output)
            else:
                output = Permute((2,3,4,1))(output)
            if num_classes==1:
                output = Activation('sigmoid', name='sigmoid'+str(i))(output)
            else:
                output = Activation(_softmax, name='softmax'+str(i))(output)
            if ndim==2:
                output_layer = Permute((3,1,2))
            else:
                output_layer = Permute((4,1,2,3))
            output_layer.name = 'output_'+str(i)
            output = output_layer(output)
            all_outputs.append(output)
    else:
        # No classifier
        all_outputs = x
    
    # MODEL
    model = Model(inputs=input, outputs=all_outputs)

    return model
