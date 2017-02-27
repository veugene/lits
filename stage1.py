# Import python libraries
from collections import OrderedDict
import numpy as np
import shutil
import os
import sys
import h5py
sys.path.append("../")

# Import keras libraries
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import (RMSprop,
                              nadam,
                              adam,
                              SGD)
from keras import backend as K
import keras

# Import in-house libraries
from callbacks import FullDice
from model.model import assemble_model
from model.blocks import (bottleneck,
                          basic_block,
                          basic_block_mp)
from fcn_plusplus.lib.loss import (masked_dice_loss,
                                   dice_loss)
from fcn_plusplus.lib.logging import FileLogger
from utils import (data_generator,
                   load_and_freeze_weights)

def train(model, num_classes, batch_size, val_batch_size, num_epochs,
          max_patience, optimizer, save_path, volume_indices, data_gen_kwargs,
          data_augmentation_kwargs={}, learning_rate=0.001, show_model=True):
    
    '''
    Metrics
    '''
    metrics = []
    
    # Accuracy
    def accuracy(y_true, y_pred):
        y_true_ = K.clip(y_true-1, 0, 1)
        if num_classes==1:
            return K.mean(K.equal(y_true, K.round(y_pred)))
        else:
            return K.mean(K.equal(K.squeeze(y_true, 1),
                                  K.argmax(y_pred, axis=1)))
    metrics.append(accuracy)
        
    # Dice averaged over slices.
    metrics.append(dice_loss(2))
    metrics.append(masked_dice_loss)

    '''
    Callbacks
    '''
    callbacks = []
    
    ## Define early stopping callback
    #early_stopping = EarlyStopping(monitor='val_acc', mode='max',
                                   #patience=max_patience, verbose=0)
    #callbacks.append(early_stopping)
    
    # Compute dice on the full data
    full_dice = FullDice()
    callbacks.append(full_dice)
    metrics.append(FullDice.get_metrics(target_class=2))
    

    # Define model saving callback
    checkpointer_best_fdice = ModelCheckpoint(filepath=os.path.join(save_path,
                                                    "best_weights_fdice.hdf5"),
                                        verbose=1,
                                        monitor='val_fdice',
                                        mode='max',
                                        save_best_only=True,
                                        save_weights_only=False)
    checkpointer_best_dice = ModelCheckpoint(filepath=os.path.join(save_path,
                                                    "best_weights_dice.hdf5"),
                                        verbose=1,
                                        monitor='val_dice',
                                        mode='min',
                                        save_best_only=True,
                                        save_weights_only=False)
    callbacks.append(checkpointer_best_fdice)
    callbacks.append(checkpointer_best_dice)
    
    # Save every last epoch
    checkpointer_last = ModelCheckpoint(filepath=os.path.join(save_path, 
                                                              "weights.hdf5"),
                                        verbose=0,
                                        save_best_only=False,
                                        save_weights_only=False)
    callbacks.append(checkpointer_last)
    
    # File logging
    logger = FileLogger(log_file_path=os.path.join(save_path,  
                                                   "training_log.txt"))
    callbacks.append(logger)
    
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
        model.compile(loss=dice_loss(2), optimizer=optimizer, metrics=metrics)

    '''
    Print model summary
    '''
    if show_model:
        from keras.utils.visualize_util import plot
        #model.summary()
        plot(model, to_file=os.path.join(save_path, 'model.png'))

    '''
    Data generators for training and validation sets
    '''
    print(' > Preparing data generators...')
    gen_train = data_generator(volume_indices=volume_indices['train'],
                               batch_size=batch_size,
                               shuffle=True,
                               loop_forever=True,
                               transform_kwargs=data_augmentation_kwargs,
                               **data_gen_kwargs)
    gen_valid = data_generator(volume_indices=volume_indices['valid'],
                               batch_size=batch_size,
                               shuffle=False,
                               loop_forever=True,
                               transform_kwargs=None,
                               **data_gen_kwargs)
    
    '''
    Train the model
    '''
    print(' > Training the model...')
    history = model.fit_generator(generator=gen_train.flow(), 
                                  samples_per_epoch=gen_train.num_samples,
                                  nb_epoch=num_epochs,
                                  validation_data=gen_valid.flow(),
                                  nb_val_samples=gen_valid.num_samples,
                                  callbacks=callbacks)
    

def main():
    '''
    Configurable parameters
    '''
    general_settings = OrderedDict((
        ('experiment_ID', "031f"),
        ('random_seed', 1234),
        ('num_train', 100),
        ('results_dir', os.path.join("/home/imagia/eugene.vorontsov-home/",
                                     "Experiments/lits/results")),
        ))

    model_kwargs = OrderedDict((
        ('input_shape', (1, 256, 256)),
        ('num_classes', 1),
        ('num_init_blocks', 2),
        ('num_main_blocks', 3),
        ('main_block_depth', 1),
        ('input_num_filters', 32),
        ('num_cycles', 1),
        ('weight_decay', 0.0005), 
        ('dropout', 0.05),
        ('batch_norm', True),
        ('mainblock', basic_block),
        ('initblock', basic_block_mp),
        ('bn_kwargs', {'momentum': 0.9, 'mode': 0}),
        ('cycles_share_weights', True),
        ('num_residuals', 2),
        ('num_first_conv', 1),
        ('num_final_conv', 1),
        ('num_classifier', 1),
        ('init', 'he_normal')
        ))

    data_gen_kwargs = OrderedDict((
        ('data_path', os.path.join("/export/projects/Candela/datasets/",
                                   "by_project/lits/data_2.zarr")),
        ('nb_io_workers', 2),
        ('nb_proc_workers', 4),
        ('downscale', True)
        ))

    data_augmentation_kwargs = OrderedDict((
        ('rotation_range', 15),
        ('width_shift_range', 0.1),
        ('height_shift_range', 0.1),
        ('shear_range', 0.),
        ('zoom_range', 0.1),
        ('channel_shift_range', 0.),
        ('fill_mode', 'constant'),
        ('cval', 0.),
        ('cvalMask', 0),
        ('horizontal_flip', True),
        ('vertical_flip', True),
        ('rescale', None),
        ('spline_warp', True),
        ('warp_sigma', 0.1),
        ('warp_grid_size', 3),
        ('crop_size', None)
        ))

    train_kwargs = OrderedDict((
        # data
        ('num_classes', 1),
        ('batch_size', 40),
        ('val_batch_size', 200),
        ('num_epochs', 500),
        ('max_patience', 50),
        
        # optimizer
        ('optimizer', 'RMSprop'),   # 'RMSprop', 'nadam', 'adam', 'sgd'
        ('learning_rate', 0.001),
        
        # other
        ('show_model', False),
        ))

    # Set random seed
    np.random.seed(general_settings['random_seed'])

    # Split data into train, validation
    num_volumes = 130
    #shuffled_indices = np.random.permutation(num_volumes)
    exclude =[32, 34, 38, 41, 47, 83, 87, 89, 91, 101, 105, 106, 114, 115, 119]
    shuffled_indices = list(set(np.arange(num_volumes)).difference(exclude))
    np.random.shuffle(shuffled_indices)
    num_train = general_settings['num_train']
    volume_indices = OrderedDict((
        ('train', shuffled_indices[:num_train]),
        ('valid', shuffled_indices[num_train:])
        ))
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
    exp = general_settings['experiment_ID']+"_"+general_settings['sub_ID']
    print("Experiment:", exp)
    print("")
    for name, d in all_dicts.items():
        print("#### {} ####".format(name))
        for key in d.keys():
            print(key, ":", d[key])
        print("")  

    '''
    Set up experiment directory
    '''
    experiment_dir = os.path.join(general_settings['results_dir'],"stage2",exp)
    model = None
    if os.path.exists(experiment_dir):
        print("")
        print("WARNING! Results directory exists: \"{}\"".format(experiment_dir))
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
        if write_into=='c':
            print("Attempting to load model state and continue training.")
            model = keras.models.load_model( \
                os.path.join(experiment_dir, "weights.hdf5"),
                custom_objects={'masked_dice_loss': masked_dice_loss,
                                'dice': dice_loss(2)})
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
        # Increase the recursion limit to handle resnet skip connections
        sys.setrecursionlimit(99999)
        model = assemble_model(**model_kwargs)
        print("   number of parameters : ", model.count_params())

        '''
        Save the model in yaml form
        '''
        yaml_string = model.to_yaml()
        open(os.path.join(experiment_dir, "model_" +
                          str(exp)+".yaml"), 'w').write(yaml_string)

    '''
    Run experiment
    '''
    train(model=model,
          save_path=experiment_dir,
          data_gen_kwargs=data_gen_kwargs,
          data_augmentation_kwargs=data_augmentation_kwargs,
          **train_kwargs)
        

if __name__ == "__main__":
    main()
