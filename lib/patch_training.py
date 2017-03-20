# Import python libraries
import numpy as np
from collections import OrderedDict
import shutil
import copy
import os
import re
import sys
import h5py

# Import keras libraries
from keras.callbacks import (EarlyStopping,
                             ModelCheckpoint,
                             CallbackList,
                             BaseLogger)
from keras.optimizers import (RMSprop,
                              nadam,
                              adam,
                              SGD)
from keras import backend as K
import keras
from keras.models import Model
from keras.layers import (Input,
                          Activation,
                          Dense,
                          BatchNormalization,
                          Convolution2D,
                          MaxPooling2D,
                          Flatten,
                          Dropout)

# Import in-house libraries
from fcn_plusplus.lib.logging import FileLogger
from fcn_plusplus.lib.resunet import _l2


def assemble_model(input_shape, num_filters,
                   num_conv_layers, dense_layer_width,
                   weight_decay=None, dropout=0., batch_norm=True,
                   bn_kwargs=None, init='he_normal'):
    
    if bn_kwargs is None:
        bn_kwargs = {}
    
    input = Input(shape=input_shape)
    x = input
    for i in range(num_conv_layers):
        x = Convolution2D(num_filters, 3, 3,
                          activation='linear',
                          init=init,
                          border_mode='same',
                          W_regularizer=_l2(weight_decay))(x)
        if batch_norm:
            x = BatchNormalization(axis=1, **bn_kwargs)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3),
                         strides=(2, 2),
                         border_mode='same')(x)
    x = Flatten()(x)
    x = Dense(dense_layer_width,
              activation='relu',
              init=init,
              W_regularizer=_l2(weight_decay))(x)
    if dropout:
        x = Dropout(dropout)(x)
    output = Dense(1, activation='sigmoid', init=init)(x)

    model = Model(input=input, output=output)
    return model


from data_tools.data_augmentation import random_transform
from data_tools.io import data_flow
from data_tools.wrap import (multi_source_array,
                             delayed_view)
import h5py
import zarr

def data_generator(data_dir, volume_indices, batch_size,
                   nb_io_workers=1, nb_proc_workers=0,
                   shuffle=False, loop_forever=False, transform_kwargs=None, data_flow_kwargs=None, rng=None, maxlen=None):
    if rng is None:
        rng = np.random.RandomState()
    
    ## Open all data files and assemble into a single multi_source_array.
    #sources = {0: [], 1: []}
    #for idx in volume_indices:
        #path = os.path.join(data_dir, "patch_set_{}.hdf5".format(idx))
        #try:
            #f = h5py.File(path, 'r')
        #except:
            #print("Failed to open data: {}".format(path))
        #sources[0].append(f['class_1'])
        #sources[1].append(f['class_2'])
    #class_list =[0]*len(sources[0]) + [1]*len(sources[1])
    #data_msa = multi_source_array(source_list=sources[0]+sources[1],
                                  #class_list=class_list,
                                  #shuffle=False)
    #data_labels = np.array(data_msa.get_labels())
    #data = [data_msa, data_labels]
    
    ## Load all data files into compressed memory.
    #data_mem = None
    #labels = []
    #for idx in volume_indices:
        #print("DEBUG: ", idx)
        #path = os.path.join(data_dir, "patch_set_{}.hdf5".format(idx))
        #try:
            #f = h5py.File(path, 'r')
        #except:
            #print("Failed to open data: {}".format(path))
        
        #if data_mem is None:
            #from zarr import blosc
            #blosc.use_threads = True
            #blosc.set_nthreads(12)
            #data_mem = zarr.array(f['class_1'][...])
            #data_mem.append(f['class_2'][...])
        #else:
            #data_mem.append(f['class_1'][...])
            #data_mem.append(f['class_2'][...])
            
        #labels.extend([0]*len(f['class_1']))
        #labels.extend([1]*len(f['class_2']))
    #data = [data_mem, np.array(labels)]
    
    # Load all data files into compressed memory.
    sources = {0: [], 1: []}
    for idx in volume_indices:
        path = os.path.join(data_dir, "patch_set_{}.hdf5".format(idx))
        try:
            f = h5py.File(path, 'r')
        except:
            print("Failed to open data: {}".format(path))
            raise
        sources[0].append(f['class_1'])
        sources[1].append(f['class_2'])
    class_list =[0]*len(sources[0]) + [1]*len(sources[1])
    data_msa = multi_source_array(source_list=sources[0]+sources[1],
                                  class_list=class_list,
                                  shuffle=False,
                                  maxlen=maxlen)
    data_labels = np.array(data_msa.get_labels())
    data_mem = data_msa[...]
    #if maxlen is not None:
        #data_msa = delayed_view(data_msa, shuffle=False, idx_max=maxlen)
        #data_labels = data_labels[:maxlen]
        
    #num_batches = len(data_msa)//batch_size
    #if len(data_msa)%batch_size:
        #num_batches += 1
    #data_mem = zarr.array(data_msa[:batch_size])
    #for i in range(num_batches):
        #sys.stdout.write("preloading {} of {}\r".format(i, num_batches))
        #sys.stdout.flush()
        #data_mem.append(data_msa[i*batch_size:(i+1)*batch_size])
    data = [data_mem, data_labels]
    
    # Function to rescale the data and do data augmentation, if requested
    def preprocessor(batch):
        b0, b1 = batch
        if transform_kwargs is not None:
            b0 = random_transform(b0, **transform_kwargs)
            #for i in range(len(b0)):
                #x = random_transform(b0[i], **transform_kwargs)
                #b0[i] = x
        # standardize
        b0 /= 255.0
        b0 = np.clip(b0, -2.0, 2.0)
        return (b0, b1)
    
    # Prepare the data iterator
    if data_flow_kwargs is None:
        data_flow_kwargs = {}
    data_gen = data_flow(data=data,
                         batch_size=batch_size,
                         nb_io_workers=nb_io_workers,
                         nb_proc_workers=nb_proc_workers,
                         shuffle=shuffle, 
                         loop_forever=loop_forever,
                         preprocessor=preprocessor,
                         rng=rng)
    return data_gen


def prepare_model(model, batch_size, val_batch_size, max_patience,
                  optimizer, save_path, volume_indices, data_gen_kwargs,
                  data_augmentation_kwargs=None, learning_rate=0.001):
    
    if data_augmentation_kwargs is None:
        data_augmentation_kwargs = {}
        
    '''
    Prepare data generators.
    '''
    gen = {}
    print(' > Preparing data generators...')
    gen['train'] = data_generator(volume_indices=volume_indices['train'],
                                  batch_size=batch_size,
                                  shuffle=True,
                                  loop_forever=True,
                                  transform_kwargs=data_augmentation_kwargs,
                                  **data_gen_kwargs)
    gen['valid'] = data_generator(volume_indices=volume_indices['valid'],
                                  batch_size=batch_size,
                                  shuffle=False,
                                  loop_forever=True,
                                  transform_kwargs=None,
                                  **data_gen_kwargs)
    
    '''
    Metrics
    '''
    metrics = ['acc']
    
    '''
    Callbacks
    '''
    callbacks = {}
    
    ## Define early stopping callback
    #early_stopping = EarlyStopping(monitor='val_acc', mode='max',
                                    #patience=max_patience, verbose=0)
    #callbacks.append(early_stopping)
    
    checkpointer_best = ModelCheckpoint(filepath=os.path.join(save_path,
                                                    "best_weights_ldice.hdf5"),
                                        verbose=1,
                                        monitor='val_acc',
                                        mode='max',
                                        save_best_only=True,
                                        save_weights_only=False)
    callbacks['checkpointer_best'] = checkpointer_best
    
    checkpointer_last = ModelCheckpoint(filepath=os.path.join(save_path,
                                                              "weights.hdf5"),
                                        verbose=0,
                                        save_best_only=False,
                                        save_weights_only=False)
    callbacks['checkpointer_last'] = checkpointer_last
    
    # File logging
    logger = FileLogger(log_file_path=os.path.join(save_path,   
                                                   "training_log.txt"))
    callbacks['logger'] = logger
    
    '''
    Compile model
    '''
    print('\n > Compiling model...')
    if optimizer=='RMSprop':
        optimizer = RMSprop(lr=learning_rate,
                            rho=0.9,
                            epsilon=1e-8,
                            decay=0.,
                            clipnorm=10)
    elif optimizer=='nadam':
        optimizer = nadam(lr=learning_rate,
                          beta_1=0.9,
                          beta_2=0.999,
                          epsilon=1e-08,
                          schedule_decay=0,
                          clipnorm=10)
    elif optimizer=='adam':
        optimizer = adam(lr=learning_rate,
                         beta_1=0.9,
                         beta_2=0.999,
                         epsilon=1e-08,
                         decay=0,
                         clipnorm=10)
    elif optimizer=='sgd':
        optimizer = SGD(lr=learning_rate,
                        momentum=0.9,
                        decay=0.,
                        nesterov=True,
                        clipnorm=10)
    else:
        raise ValueError("Unknown optimizer: {}".format(optimizer))
    if not hasattr(model, 'optimizer'):
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=metrics)
        
    return model, callbacks, gen
        

def train(model, num_epochs, initial_epoch=0, **kwargs):
    model, callbacks, gen = prepare_model(model=model, **kwargs)
    print(' > Training the model...')
    history = model.fit_generator(generator=gen['train'].flow(),
                                  samples_per_epoch=5000000,
                                  nb_epoch=num_epochs,
                                  validation_data=gen['valid'].flow(),
                                  nb_val_samples=gen['valid'].num_samples,
                                  callbacks=list(callbacks.values()),
                                  initial_epoch=initial_epoch)
    return history


def load_model(path):
    custom_object_list = []
    # nothing here...
    custom_objects = dict((f.__name__, f) for f in custom_object_list)
    model = keras.models.load_model(path, custom_objects=custom_objects)
    return model


def run(general_settings,
        model_kwargs,
        data_gen_kwargs,
        data_augmentation_kwargs,
        train_kwargs,
        loader_kwargs=None):
    
    # Set random seed
    np.random.seed(general_settings['random_seed'])
    
    # Split data into train, validation
    num_volumes = 130
    shuffled_indices = np.arange(num_volumes)
    if general_settings['exclude_data'] is not None:
        exclude = general_settings['exclude_data']
        shuffled_indices = list(set(shuffled_indices).difference(exclude))
    np.random.shuffle(shuffled_indices)
    num_train = general_settings['num_train']
    volume_indices = OrderedDict()
    volume_indices['train'] = shuffled_indices[:num_train]
    if 'validation_set' in general_settings and \
        general_settings['validation_set'] is not None:
        volume_indices['valid'] = general_settings['validation_set']
    else:
        volume_indices['valid'] = shuffled_indices[num_train:]
    train_kwargs.update({'volume_indices': volume_indices})

    '''
    Print settings to screen
    '''
    all_dicts = OrderedDict((
        ("General settings", general_settings),
        ("Model settings", model_kwargs),
        ("Data generator settings", data_gen_kwargs),
        ("Data augmentation settings", data_augmentation_kwargs),
        ("Trainer settings", train_kwargs)
        ))
    print("Experiment:", general_settings['save_subdir'])
    print("")
    for name, d in all_dicts.items():
        print("#### {} ####".format(name))
        for key in d.keys():
            print(key, ":", d[key])
        print("")  

    '''
    Set up experiment directory
    '''
    experiment_dir = os.path.join(general_settings['results_dir'],
                                  general_settings['save_subdir'])
    model = None
    initial_epoch = 0
    if os.path.exists(experiment_dir):
        print("")
        print("WARNING! Results directory exists: \"{}\""
              "".format(experiment_dir))
        write_into = None
        while write_into not in ['y', 'n', 'r', 'c', '']:
            write_into = str.lower(input( \
                            "Write into existing directory?\n"
                            "    y : yes\n"
                            "    n : no (default)\n"
                            "    r : delete and replace directory\n"
                            "    c : continue/resume training\n"))
        if write_into in ['n', '']:
            print("Aborted")
            sys.exit()
        if write_into=='r':
            print("WARNING: Deleting existing results directory.")
            shutil.rmtree(experiment_dir)
        if write_into in ['c']:
            print("Loading model.")
            model = load_model(path=os.path.join(experiment_dir,
                                                 "weights.hdf5"))
            
            # Identify initial epoch
            log_path = os.path.join(experiment_dir, "training_log.txt")
            if os.path.exists(log_path):
                f = open(log_path, 'r')
                last_line = f.readlines()[-1]
                last_epoch = int(re.split('[: ]+', last_line)[1])
                initial_epoch = last_epoch-1
        print("")
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
        
        
    '''
    Save this experiment script in the experiment directory
    '''
    fn = sys.argv[0].rsplit('/', 1)[-1]
    shutil.copy(sys.argv[0], os.path.join(experiment_dir, fn))

    '''
    Assemble model
    '''
    if model is None:
        print('\n > Building model...')
        model = assemble_model(**model_kwargs)
        print("   number of parameters : ", model.count_params())

        #'''
        #Save the model in yaml form
        #'''
        #yaml_string = model.to_yaml()
        #open(os.path.join(experiment_dir,"model.yaml"), 'w').write(yaml_string)
        
        '''
        Load model weights.
        '''
        if general_settings['load_subpath'] is not None:
            load_path = os.path.join(general_settings['results_dir'],
                                     general_settings['load_subpath'])
            model.load_weights(load_path)
            

    '''
    Run experiment
    '''
    train(model=model,
          save_path=experiment_dir,
          data_gen_kwargs=data_gen_kwargs,
          data_augmentation_kwargs=data_augmentation_kwargs,
          initial_epoch=initial_epoch,
          **train_kwargs)
 
