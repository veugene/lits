# Import python libraries
import numpy as np
from collections import OrderedDict
import shutil
import copy
import os
import re
import sys
import h5py
sys.path.append("../")

# Import keras libraries
from keras.callbacks import (EarlyStopping,
                             ModelCheckpoint,
                             CallbackList,
                             BaseLogger)
from keras.optimizers import (RMSprop,
                              nadam,
                              adam,
                              SGD)
from keras.losses import mean_squared_error
from keras import backend as K
import keras

# Import in-house libraries
from .callbacks import (Dice,
                        SavePredictions,
                        FileLogger)
from .model_variants import assemble_model
from .normalization_layers import (WeightNorm,
                                   LayerNorm)
from .loss import dice_loss
from .utils import (data_generator,
                    repeat_flow,
                    load_and_freeze_weights)
    
    
def prepare_model(model, num_classes, batch_size, val_batch_size, max_patience,
                  optimizer, save_path, volume_indices, data_gen_kwargs,
                  data_augmentation_kwargs=None, learning_rate=0.001,
                  num_outputs=1, adversarial=False, adv_weight=0.2,
                  save_every=0, mask_to_liver=False, show_model=True,
                  liver_only=False):
    
    if data_augmentation_kwargs is None:
        data_augmentation_kwargs = {}
    
    if num_outputs not in [1, 2]:
        raise ValueError("num_outputs must be 1 or 2")
    
    if liver_only and num_outputs!=1:
        raise ValueError("num_outputs must be 1 when liver_only is True")
    
    if not liver_only:
        lesion_output = 'output_0'
        liver_output = 'output_1'
    else:
        lesion_output = None
        liver_output = 'output_0'
        
    if adversarial and not num_outputs==2:
        print("num_outputs must be 2 when adversarial is True")
    
    '''
    Data generators for training and validation sets
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
    gen['valid_callback'] = data_generator( \
                                        volume_indices=volume_indices['valid'],
                                        batch_size=batch_size,
                                        shuffle=False,
                                        loop_forever=False,
                                        transform_kwargs=None,
                                        **data_gen_kwargs)
    
    
    '''
    Metrics
    '''
    metrics = {}
    if lesion_output:
        metrics[lesion_output] = []
    if num_outputs==2 or lesion_output is None:
        metrics[liver_output] = []
        
    # Accuracy
    def accuracy(y_true, y_pred):
        y_true_ = K.clip(y_true-1, 0, 1)
        if num_classes==1:
            return K.mean(K.equal(y_true, K.round(y_pred)))
        else:
            return K.mean(K.equal(K.squeeze(y_true, 1),
                                  K.argmax(y_pred, axis=1)))
    if lesion_output:
        metrics[lesion_output].append(accuracy)
        
    # Dice averaged over slices.
    if lesion_output:
        metrics[lesion_output].append(dice_loss(2))
        metrics[lesion_output].append(dice_loss(2, masked_class=0))
    if num_outputs==2 or lesion_output is None:
        metrics[liver_output].append(dice_loss([1, 2]))

    '''
    Callbacks
    '''
    callbacks = OrderedDict()
    
    ## Define early stopping callback
    #early_stopping = EarlyStopping(monitor='val_acc', mode='max',
                                   #patience=max_patience, verbose=0)
    #callbacks.append(early_stopping)
    
    # Save prediction images
    if save_every:
        save_predictions = SavePredictions(num_epochs=save_every,
                                           data_gen=gen['valid_callback'],
                                           save_path=os.path.join(save_path,
                                               "predictions"))
        callbacks['save_predictions'] = save_predictions
    
    # Compute dice on the full data
    if lesion_output:
        output_name = lesion_output if num_outputs==2 else None
        dice_lesion = Dice(target_class=2, output_name=output_name)
        dice_lesion_inliver = Dice(target_class=2, mask_class=0,
                                   output_name=output_name)
        callbacks['dice_lesion'] = dice_lesion
        callbacks['dice_lesion_inliver'] = dice_lesion_inliver
        metrics[lesion_output].extend(dice_lesion.get_metrics())
        metrics[lesion_output].extend(dice_lesion_inliver.get_metrics())
    if num_outputs==2 or lesion_output is None:
        output_name = liver_output if num_outputs==2 else None
        dice_liver = Dice(target_class=[1, 2], output_name=output_name)
        callbacks['dice_liver'] = dice_liver
        metrics[liver_output].extend(dice_liver.get_metrics())
    

    # Define model saving callback
    if lesion_output is not None:
        monitor = 'val_dice_loss_2' if num_outputs==1 \
            else 'val_output_0_dice_loss_2'
        if mask_to_liver:
            monitor += '_m0'
    else:
        monitor = 'val_dice_loss_1_2' if num_outputs==1 \
            else 'val_output_0_dice_loss_1_2'
    checkpointer_best_ldice = ModelCheckpoint(filepath=os.path.join(save_path,
                                                    "best_weights_ldice.hdf5"),
                                              verbose=1,
                                              monitor=monitor,
                                              mode='min',
                                              save_best_only=True,
                                              save_weights_only=False)
    if lesion_output is not None:
        monitor = 'val_dice_2' if num_outputs==1 else 'val_output_0_dice_2'
        if mask_to_liver:
            monitor += '_m0'
    else:
        monitor = 'val_dice_1_2' if num_outputs==1 \
            else 'val_output_0_dice_1_2'
    checkpointer_best_dice = ModelCheckpoint(filepath=os.path.join(save_path,
                                                    "best_weights_dice.hdf5"),
                                             verbose=1,
                                             monitor=monitor,
                                             mode='max',
                                             save_best_only=True,
                                             save_weights_only=False)
    callbacks['checkpointer_best_ldice'] = checkpointer_best_ldice
    callbacks['checkpointer_best_dice'] = checkpointer_best_dice
    
    
    # Save every last epoch
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
        def loss(loss_func, weight):
            def f(y_true, y_pred):
                loss = loss_func(y_true, y_pred)*weight
                loss.__name__ = loss_func.__name__
                return loss
            return f
        masked_class = 0 if mask_to_liver else None
        losses = {}
        if lesion_output:
            losses[lesion_output] = loss(dice_loss(2, masked_class=masked_class),
                                         1.-adv_weight)
        if num_outputs==2 or lesion_output is None:
            losses[liver_output] = loss(dice_loss([1, 2]), 1.-adv_weight)
        if adversarial:
            losses['out_adv_0_d'] = loss(mean_squared_error, adv_weight)
            losses['out_adv_1_d'] = loss(mean_squared_error, adv_weight)
            losses['out_adv_0_g'] = loss(mean_squared_error, adv_weight)
            losses['out_adv_1_g'] = loss(mean_squared_error, adv_weight)
            losses['out_disc_0'] = loss(mean_squared_error, adv_weight)
            losses['out_disc_1'] = loss(mean_squared_error, adv_weight)
        model.compile(loss=losses, optimizer=optimizer, metrics=metrics)
        
    '''
    Print model summary
    '''
    if show_model:
        from keras.utils.visualize_util import plot
        #model.summary()
        plot(model, to_file=os.path.join(save_path, 'model.png'))

    return model, callbacks, gen
    
    
def train(model, num_epochs, num_outputs,
          adversarial=False, adversarial_weight=0.2, initial_epoch=0,
          **kwargs):
    model, callbacks, gen = prepare_model(model=model,
                                          num_outputs=num_outputs,
                                          adversarial=adversarial,
                                          adv_weight=adversarial_weight,
                                          **kwargs)
    
    # Duplicate inputs and outputs (and add outputs) as necessary.
    flow = {}
    for key in gen:
        flow[key] = repeat_flow(gen[key].flow(),
                                num_outputs=num_outputs,
                                adversarial=adversarial)
    
    print(' > Training the model...')
    history = model.fit_generator(generator=flow['train'],
                                  steps_per_epoch=len(gen['train']),
                                  epochs=num_epochs,
                                  validation_data=flow['valid'],
                                  validation_steps=len(gen['valid']),
                                  callbacks=list(callbacks.values()),
                                  initial_epoch=initial_epoch)
    return history
    

def evaluate(model, save_path, num_outputs, liver_only=False, **kwargs):
    model, callbacks, gen = prepare_model(model=model,
                                          save_path=save_path,
                                          num_outputs=num_outputs,
                                          liver_only=liver_only,
                                          **kwargs)
    
    print(' > Evaluating the model...')
    from scipy.misc import imsave
    
    # Create directory, if needed
    save_predictions_to = os.path.join(save_path, "predictions")
    if not os.path.exists(save_predictions_to):
        os.makedirs(save_predictions_to)
        
    # Initialize callbacks
    val_callback_list = [BaseLogger()]
    if not liver_only:
        val_callback_list.extend([callbacks['dice_lesion'],
                                  callbacks['dice_lesion_inliver']])
    if len(model.outputs)==2 or liver_only:
        val_callback_list.append(callbacks['dice_liver'])
    val_callbacks = CallbackList(val_callback_list)
    val_callbacks.set_params({
        'nb_epoch': 0,
        'nb_sample': 0,
        'verbose': False,
        'do_validation': True,
        'metrics': model.metrics_names})
    val_callbacks.on_train_begin()
    val_callbacks.on_epoch_begin(0)
    
    # Create theano function
    inputs = model.inputs + model.targets + model.sample_weights
    if model.uses_learning_phase and \
            not isinstance(K.learning_phase(), int):
        inputs += [K.learning_phase()]
    predict_and_test_function = K.function( \
        inputs,
        model.outputs+[model.total_loss]+model.metrics_tensors,
        updates=model.state_updates)
    
    # Loop through batches, applying function and callbacks
    flow = repeat_flow(gen['valid_callback'].flow(), num_outputs=num_outputs)
    for batch_num, batch in enumerate(flow):
        x, y, sample_weights = model._standardize_user_data(batch[0],
                                                            batch[1])
        ins = x+y+sample_weights
        if model.uses_learning_phase and \
                not isinstance(K.learning_phase(), int):
            ins += [0.]
        outputs = predict_and_test_function(ins)
        if num_outputs==1:
            predictions = outputs[0:1]
            val_metrics = outputs[1:]
        else:
            predictions = outputs[0:2]
            val_metrics = outputs[2:]
        
        # Save images
        def process_slice(s):
            s = np.squeeze(s).copy()
            s[s<0]=0
            s[s>1]=1
            s[0,0]=1
            s[0,1]=0
            return s
        for i in range(len(batch[0])):
            s_pred_list = []
            if num_outputs==1:
                s_pred_list = [process_slice(predictions[i])]
            else:
                for j in range(num_outputs):
                    s_pred_list.append(process_slice(predictions[j][i]))
            s_input = process_slice(batch[0][i])
            if num_outputs==1:
                s_truth = process_slice(batch[1][i]/2.)
            else:
                s_truth = process_slice(batch[1][0][i]/2.)
            out_image = np.concatenate([s_input]+s_pred_list+[s_truth],
                                        axis=1)
            imsave(os.path.join(save_predictions_to,
                                "{}_{}.png".format(batch_num, i)),
                    out_image)
            
        # Update metrics
        val_logs = OrderedDict(zip(model.metrics_names, val_metrics))
        val_logs.update({'batch': batch_num, 'size': len(batch[0])})
        val_callbacks.on_batch_end(batch_num, val_logs)
    
    # Update metrics
    val_callbacks.on_epoch_end(0, val_logs)
    
    # Output metrics
    for m in val_logs:
        if m not in ['batch', 'size']:
            print("{}: {}".format(m, val_logs[m]))
            
            
def load_model(path, num_outputs, liver_only):
    custom_object_list = []    
    if not liver_only:
        les_output_name = None
        if num_outputs==2:
            les_output_name = 'output_0'
        custom_object_list.append(Dice(2, output_name=les_output_name))
        custom_object_list.extend(custom_object_list[-1].get_metrics())
        custom_object_list.append(Dice(2, mask_class=0,
                                       output_name=les_output_name))
        custom_object_list.extend(custom_object_list[-1].get_metrics())
        custom_object_list.append(dice_loss(2))
        custom_object_list.append(dice_loss(2, masked_class=0))
    if num_outputs==2 or liver_only:
        liv_output_name = 'output_0'
        if num_outputs==2:
            liv_output_name = 'output_1'
        custom_object_list.append(Dice([1, 2], output_name=liv_output_name))
        custom_object_list.extend(custom_object_list[-1].get_metrics())
        custom_object_list.append(dice_loss([1, 2]))
    custom_objects = dict((f.__name__, f) for f in custom_object_list)
    custom_objects['WeightNorm'] = WeightNorm
    custom_objects['LayerNorm'] = LayerNorm
    model = keras.models.load_model(path, custom_objects=custom_objects)
    return model


def run(general_settings,
        model_kwargs,
        data_gen_kwargs,
        data_augmentation_kwargs,
        train_kwargs,
        discriminator_kwargs=None,
        loader_kwargs=None):
    
    # Set random seed
    np.random.seed(general_settings['random_seed'])
    
    # Increase the recursion limit to handle resnet skip connections
    sys.setrecursionlimit(99999)

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
        
    if 'adversarial' in model_kwargs:
        adversarial = model_kwargs['adversarial']
    else:
        adversarial = False

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
            liver_only = False
            if 'liver_only' in train_kwargs and train_kwargs['liver_only']:
                liver_only = True
            model = load_model(path=os.path.join(experiment_dir,
                                                 "weights.hdf5"),
                               num_outputs=model_kwargs['num_outputs'],
                               liver_only=liver_only)
            
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
        if discriminator_kwargs is None:
            model = assemble_model(**model_kwargs)
        else:
            model = assemble_model(discriminator_kwargs=discriminator_kwargs,
                                   **model_kwargs)
        print("   number of parameters : ", model.count_params())

        '''
        Save the model in yaml form
        '''
        yaml_string = model.to_yaml()
        open(os.path.join(experiment_dir,"model.yaml"), 'w').write(yaml_string)
        
        '''
        Load model weights and freeze some of them.
        '''
        if general_settings['load_subpath'] is not None:
            load_path = os.path.join(general_settings['results_dir'],
                                     general_settings['load_subpath'])
            if loader_kwargs is None:
                loader_kwargs = {}
                for key in ['freeze', 'layers_to_not_freeze']:
                    # Backward compatibility
                    if key in general_settings:
                        loader_kwargs[key] = general_settings[key]
            load_and_freeze_weights(model, load_path, verbose=True,
                                    **loader_kwargs)
            #model.save(os.path.join(experiment_dir, "model.hdf5"))
            #model.load_weights(load_path)
            #model.save_weights(load_path+'.renamed')
            #sys.exit()
            
    '''
    Evaluate
    '''
    if 'evaluate' in general_settings and general_settings['evaluate']:
        print("Evaluating model on validation set.")
        evaluate_kwargs = dict([(kw, train_kwargs[kw]) for kw in train_kwargs \
            if kw not in ['num_epochs', 'initial_epoch']])
        evaluate(model=model,
                save_path=os.path.join(experiment_dir, "predictions"),
                data_gen_kwargs=data_gen_kwargs,
                **evaluate_kwargs)
        sys.exit()

    '''
    Run experiment
    '''
    train(model=model,
          save_path=experiment_dir,
          data_gen_kwargs=data_gen_kwargs,
          data_augmentation_kwargs=data_augmentation_kwargs,
          initial_epoch=initial_epoch,
          adversarial=adversarial,
          **train_kwargs)
