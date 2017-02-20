from keras.models import Model
from keras.layers import (Input,
                          Activation,
                          Dense,
                          Permute,
                          Lambda,
                          merge)
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras import backend as K
from theano import tensor as T
from keras.regularizers import l2
import numpy as np
from fcn_plusplus.lib.blocks import (bottleneck,
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
    
    
def assemble_model(input_shape, num_classes, num_init_blocks, num_main_blocks,
                   main_block_depth, input_num_filters, num_cycles=1,
                   preprocessor_network=None, postprocessor_network=None,
                   mainblock=None, initblock=None, firstblock=None,
                   dropout=0., batch_norm=True, weight_decay=None,
                   cycles_share_weights=True):
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
    firstblock : a layer defining the firstblock (basic_block_mp by default).
    use_skip_blocks : pass features skipped along long_skip through skipblocks.
    dropout : the dropout probability, introduced in every block.
    batch_norm : enable or disable batch normalization.
    weight_decay : the weight decay (L2 penalty) used in every convolution.
    cycles_share_weights : share network weights across cycles.
    """
    
    '''
    By default, use depth 2 basic_block for mainblock
    '''
    if mainblock is None:
        mainblock = basic_block
    if initblock is None:
        initblock = basic_block_mp
    if firstblock is None:
        firstblock = basic_block_mp
    
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
    '''
    def merge_in(x, into):
        if x._keras_shape[1] != into._keras_shape[1]:
            x = Convolution2D(into._keras_shape[1], 1, 1,
                              init='he_normal', border_mode='valid',
                              W_regularizer=_l2(weight_decay))(x)
        out = merge([x, into], mode='sum')
        return out
    
    '''
    Constant kwargs passed to the init and main blocks.
    '''
    block_kwargs = {'dropout': dropout,
                    'batch_norm': batch_norm,
                    'weight_decay': weight_decay}
    
    # INPUT
    input = Input(shape=input_shape)
    
    # Preprocessing
    if preprocessor_network is not None:
        input = preprocessor_network(input)
    
    '''
    Build the blocks for all cycles, contracting and expanding in each cycle.
    '''
    tensors = []
    layers = []
    x = input
    for cycle in range(num_cycles):
        # Create tensors and layer lists for this cycle.
        tensors.append({'down': {}, 'up': {}, 'across': {}})
        layers.append({'down': {}, 'up': {}, 'across': {}})
        
        # First block
        if cycle > 0:
            x = merge_in(x, tensors[cycle-1]['up'][0])
        if cycles_share_weights and cycle > 0:
            block = layers[cycle-1]['down'][0]
        else:
            block = firstblock(nb_filter=input_num_filters,
                               subsample=False,
                               upsample=False,
                               skip=False,
                               **block_kwargs)
        x = block(x)
        layers[cycle]['down'][0] = block
        tensors[cycle]['down'][0] = x
        print("Cycle {} - FIRST DOWN: {}".format(cycle, x._keras_shape))
        
        # DOWN (initial subsampling blocks)
        for b in range(num_init_blocks):
            depth = b+1
            if cycle > 0:
                x = merge_in(x, tensors[cycle-1]['up'][depth])
            if cycles_share_weights and cycle > 0:
                block = layers[cycle-1]['down'][depth]
            else:
                block = residual_block(initblock,
                                       nb_filter=input_num_filters,
                                       repetitions=1,
                                       skip=True,
                                       subsample=True,
                                       upsample=False,
                                       **block_kwargs)
            x = block(x)
            layers[cycle]['down'][depth] = block
            tensors[cycle]['down'][depth] = x
            print("Cycle {} - INIT DOWN {}: {}".format(cycle, b,
                                                       x._keras_shape))
        
        # DOWN (resnet blocks)
        for b in range(num_main_blocks):
            depth = b+1+num_init_blocks
            if cycle > 0:
                x = merge_in(x, tensors[cycle-1]['up'][depth])
            if cycles_share_weights and cycle > 0:
                block = layers[cycle-1]['down'][depth]
            else:
                block = residual_block(mainblock,
                                       nb_filter=input_num_filters*(2**b),
                                       repetitions=get_repetitions(b),
                                       skip=True,
                                       subsample=True,
                                       upsample=False,
                                       **block_kwargs)
            x = block(x)
            layers[cycle]['down'][depth] = block
            tensors[cycle]['down'][depth] = x
            print("Cycle {} - MAIN DOWN {}: {}".format(cycle, b,
                                                       x._keras_shape))
            
        # ACROSS
        if cycles_share_weights and cycle > 0:
            block = layers[cycle-1]['across'][0]
        else:
            block = residual_block(mainblock,
                                  nb_filter=input_num_filters*(2**b),
                                  repetitions=get_repetitions(num_main_blocks),
                                  skip=True,
                                  subsample=True,
                                  upsample=True,
                                  **block_kwargs)
        x = block(x)
        if cycle > 0:
            x = merge_in(x, tensors[cycle-1]['across'][0])
        layers[cycle]['across'][0] = block
        tensors[cycle]['across'][0] = x
        print("Cycle {} - ACROSS: {}".format(cycle, x._keras_shape))

        # UP (resnet blocks)
        for b in range(num_main_blocks-1, -1, -1):
            depth = b+1+num_init_blocks
            x = merge_in(x, tensors[cycle]['down'][depth])
            if cycles_share_weights and cycle > 0:
                block = layers[cycle-1]['up'][depth]
            else:
                block = residual_block(mainblock,
                                       nb_filter=input_num_filters*(2**b),
                                       repetitions=get_repetitions(b),
                                       skip=True,
                                       subsample=False,
                                       upsample=True,
                                       **block_kwargs)
            x = block(x)
            layers[cycle]['up'][depth] = block
            tensors[cycle]['up'][depth] = x
            print("Cycle {} - MAIN UP {}: {}".format(cycle, b,
                                                     x._keras_shape))
        
        # UP (final upsampling blocks)
        for b in range(num_init_blocks-1, -1, -1):
            depth = b+1
            x = merge_in(x, tensors[cycle]['down'][depth])
            if cycles_share_weights and cycle > 0:
                block = layers[cycle-1]['up'][depth]
            else:
                block = residual_block(initblock,
                                       nb_filter=input_num_filters,
                                       repetitions=1,
                                       skip=True,
                                       subsample=False,
                                       upsample=True,
                                       **block_kwargs)
            x = block(x)
            layers[cycle]['up'][depth] = block
            tensors[cycle]['up'][depth] = x
            print("Cycle {} - INIT UP {}: {}".format(cycle, b,
                                                     x._keras_shape))
            
        # Final block
        x = merge_in(x, tensors[cycle]['down'][0])
        if cycles_share_weights and cycle > 0:
            block = layers[cycle-1]['up'][0]
        else:
            block = firstblock(nb_filter=input_num_filters,
                               subsample=False,
                               upsample=False,
                               skip=False,
                               **block_kwargs)
        x = block(x)
        layers[cycle]['up'][0] = block
        tensors[cycle]['up'][0] = x
        print("Cycle {} - FIRST UP: {}".format(cycle, x._keras_shape))
            
    # Postprocessing
    if postprocessor_network is not None:
        x = postprocessor_network(x)
    
    # OUTPUT (SOFTMAX)
    if num_classes is not None:
        # Linear classifier
        output = Convolution2D(num_classes,1,1,activation='linear', 
                               W_regularizer=_l2(weight_decay))(x)
        output = Permute((2,3,1))(output)
        if num_classes==1:
            output = Activation('sigmoid')(output)
        else:
            output = Activation(_softmax)(output)
        output = Permute((3,1,2))(output)
    else:
        # No classifier
        output = x
    
    # MODEL
    model = Model(input=input, output=output)

    return model
