# Import python libraries
import numpy as np
from collections import OrderedDict
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
from .callbacks import (Dice,
                        SavePredictions)
from .model import assemble_model
from .loss import dice_loss
from fcn_plusplus.lib.logging import FileLogger
from .utils import (data_generator,
                    repeat_flow,
                    load_and_freeze_weights)

def train(model, num_classes, batch_size, val_batch_size, num_epochs,
          max_patience, optimizer, save_path, volume_indices, data_gen_kwargs,
          data_augmentation_kwargs={}, learning_rate=0.001, num_outputs=1,
          save_every=0, show_model=True):
    
    if num_outputs not in [1, 2]:
        raise ValueError("num_outputs must be 1 or 2")
    
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
    gen_train_flow = repeat_flow(gen_train.flow(), num_outputs=num_outputs)
    gen_valid_flow = repeat_flow(gen_valid.flow(), num_outputs=num_outputs)
    
    '''
    Metrics
    '''
    metrics = {'output_0': []}
    if num_outputs==2:
        metrics['output_1'] = []
        
    # Accuracy
    def accuracy(y_true, y_pred):
        y_true_ = K.clip(y_true-1, 0, 1)
        if num_classes==1:
            return K.mean(K.equal(y_true, K.round(y_pred)))
        else:
            return K.mean(K.equal(K.squeeze(y_true, 1),
                                  K.argmax(y_pred, axis=1)))
    metrics['output_0'].append(accuracy)
        
    # Dice averaged over slices.
    metrics['output_0'].append(dice_loss(2))
    metrics['output_0'].append(dice_loss(2, masked_class=0))
    if num_outputs==2:
        metrics['output_1'].append(dice_loss([1, 2]))

    '''
    Callbacks
    '''
    callbacks = []
    
    ## Define early stopping callback
    #early_stopping = EarlyStopping(monitor='val_acc', mode='max',
                                   #patience=max_patience, verbose=0)
    #callbacks.append(early_stopping)
    
    # Save prediction images
    if save_every:
        save_predictions = SavePredictions(num_epochs=save_every,
                                           data_gen=gen_valid,
                                           save_path=os.path.join(save_path,
                                               "predictions"))
        callbacks.append(save_predictions)
    
    # Compute dice on the full data
    output_name = 'output_0' if num_outputs==2 else None
    dice_lesion = Dice(target_class=2, output_name=output_name)
    dice_lesion_inliver = Dice(target_class=2, mask_class=0,
                               output_name=output_name)
    callbacks.extend([dice_lesion, dice_lesion_inliver])
    metrics['output_0'].extend([dice_lesion.get_metrics(),
                                dice_lesion_inliver.get_metrics()])
    if num_outputs==2:
        dice_liver = Dice(target_class=[1, 2], output_name='output_1')
        callbacks.append(dice_liver)
        metrics['output_1'].append(dice_liver.get_metrics())
    

    # Define model saving callback
    monitor = 'val_dice_2' if num_outputs==1 else 'output_0_val_dice_2'
    checkpointer_best_dice = ModelCheckpoint(filepath=os.path.join(save_path,
                                                    "best_weights_dice.hdf5"),
                                        verbose=1,
                                        monitor=monitor,
                                        mode='max',
                                        save_best_only=True,
                                        save_weights_only=False)
    monitor = 'val_dice_loss_2' if num_outputs==1 \
        else 'output_0_val_dice_loss_2'
    checkpointer_best_ldice = ModelCheckpoint(filepath=os.path.join(save_path,
                                                    "best_weights_ldice.hdf5"),
                                        verbose=1,
                                        monitor=monitor,
                                        mode='min',
                                        save_best_only=True,
                                        save_weights_only=False)
    callbacks.append(checkpointer_best_dice)
    callbacks.append(checkpointer_best_ldice)
    
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
        losses = {'output_0': dice_loss(2)}
        if num_outputs==2:
            losses['output_1'] = dice_loss([1, 2])
        model.compile(loss=losses, optimizer=optimizer, metrics=metrics)
        
    '''
    Print model summary
    '''
    if show_model:
        from keras.utils.visualize_util import plot
        #model.summary()
        plot(model, to_file=os.path.join(save_path, 'model.png'))
    
    '''
    Train the model
    '''
    print(' > Training the model...')
    history = model.fit_generator(generator=gen_train_flow,
                                  samples_per_epoch=gen_train.num_samples,
                                  nb_epoch=num_epochs,
                                  validation_data=gen_valid_flow,
                                  nb_val_samples=gen_valid.num_samples,
                                  callbacks=callbacks)


def run(general_settings,
        model_kwargs,
        data_gen_kwargs,
        data_augmentation_kwargs,
        train_kwargs):
    
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
            custom_object_list = []
            output_name = 'output_0' if num_outputs==2 else None 
            custom_object_list.append(Dice(2), output_name=output_name)
            custom_object_list.append(custom_object_list[-1].get_metrics())
            custom_object_list.append(Dice(2, masked_class=0), 
                                        output_name=output_name)
            custom_object_list.append(custom_object_list[-1].get_metrics())
            custom_object_list.append(Dice([1, 2]), output_name='output_1')
            custom_object_list.append(custom_object_list[-1].get_metrics())
            custom_object_list.append(dice_loss(2))
            custom_object_list.append(dice_loss(2, masked_class=0))
            custom_object_list.append(dice_loss([1, 2]))
            custom_objects = dict((f.__name__, f) for f in custom_object_list)
            model = keras.models.load_model( \
                os.path.join(experiment_dir, "weights.hdf5"),
                custom_objects=custom_objects)
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
        open(os.path.join(experiment_dir,"model.yaml"), 'w').write(yaml_string)
        
        '''
        Load model weights and freeze some of them.
        '''
        load_path = os.path.join(general_settings['results_dir'],
                                 general_settings['load_subpath'])
        load_and_freeze_weights(model, load_path,
                 freeze=general_settings['freeze'],
                 layers_to_not_freeze=general_settings['layers_to_not_freeze'],
                 verbose=True)

    '''
    Run experiment
    '''
    train(model=model,
          save_path=experiment_dir,
          data_gen_kwargs=data_gen_kwargs,
          data_augmentation_kwargs=data_augmentation_kwargs,
          **train_kwargs)