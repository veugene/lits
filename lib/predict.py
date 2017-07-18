# Import python libraries
import numpy as np
from collections import OrderedDict
import shutil
import copy
import os
import re
import sys
import h5py
import zarr
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
from keras.engine.training import _standardize_input_data
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
from .utils import (volume_generator,
                    repeat_flow,
                    load_and_freeze_weights)


def prepare_model(model, num_classes, volume_indices, data_gen_kwargs,
                  num_outputs=1,  mask_to_liver=False, show_model=True,
                  liver_only=False, evaluate=False):
    
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
        
    '''
    Data generators for training and validation sets
    '''
    gen = {}
    print(' > Preparing data generators...')
    gen['train'] = volume_generator(volume_indices=volume_indices['train'],
                                    return_vol_idx=True,
                                    **data_gen_kwargs)
    gen['valid'] = volume_generator(volume_indices=volume_indices['valid'],
                                    return_vol_idx=True,
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
    
    '''
    Compile model
    '''
    print('\n > Compiling model...')
    if not hasattr(model, 'optimizer'):
        masked_class = 0 if mask_to_liver else None
        losses = {}
        if lesion_output:
            losses[lesion_output] = dice_loss(2, masked_class=masked_class)
        if num_outputs==2 or lesion_output is None:
            losses[liver_output] = dice_loss([1, 2])
        model.compile(loss=losses, optimizer='RMSprop', metrics=metrics)
        
    '''
    Print model summary
    '''
    if show_model:
        from keras.utils.visualize_util import plot
        #model.summary()
        plot(model, to_file=os.path.join(save_path, 'model.png'))

    return model, callbacks, gen
    
    
def predict(model, batch_size, num_outputs, save_path,
            evaluate=False, liver_only=False, save_predictions=False,
            initial_epoch=0, **kwargs):
    model, callbacks, gen = prepare_model(model=model,
                                          num_outputs=num_outputs,
                                          liver_only=liver_only,
                                          evaluate=evaluate,
                                          **kwargs)
    
    # Set up prediction file.
    if save_predictions:
        save_path = os.path.join(save_path, "predictions.zarr")
        if os.path.exists(save_path):
            os.remove(save_path)
        
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
    if evaluate:
        inputs = model.inputs + model.targets + model.sample_weights
        if model.uses_learning_phase and \
                not isinstance(K.learning_phase(), int):
            inputs += [K.learning_phase()]
        predict_function = K.function(
            inputs,
            model.outputs+[model.total_loss]+model.metrics_tensors,
            updates=model.state_updates)
    else:
        inputs = model.inputs
        if model.uses_learning_phase and \
                not isinstance(K.learning_phase(), int):
            inputs += [K.learning_phase()]  
        predict_function = K.function(
            inputs,
            model.outputs,
            updates=model.state_updates)
    
    # Predict for all data.    
    print(' > Predicting...')
    for key in gen:
        print(' - DATA: {}'.format(key))
        
        # Duplicate inputs and outputs (and add outputs) as necessary.
        flow = repeat_flow(gen[key].flow(),
                           num_outputs=num_outputs)
        
        # Set up file.
        if save_predictions:
            zgroup = zarr.open_group(store=save_path, mode='a', path="/")
            zarr_kwargs = {'chunks': (1, 512, 512),
                        'compressor': zarr.Blosc(cname='lz4',
                                                    clevel=9, shuffle=1)}
        
        # Predict and write to file.
        batch_num = 0
        for vol_num, volume in enumerate(flow):
            print("Predicting on `{}` - {}/{}"
                  "".format(key, vol_num+1, len(gen[key])))
            
            # Begin writing to file.
            if save_predictions:
                vol_idx = volume[-1]
                subgroup = zgroup.create_group(str(vol_idx))
                num_channels = np.sum(model.output_shape[i][1] \
                                                   for i in range(num_outputs))
                output_shape = \
                       (len(volume[0]), num_channels)+model.output_shape[0][2:]
                subgroup.empty("volume",
                               shape=output_shape,
                               dtype=np.float32, **zarr_kwargs)
                segmentation = volume[1]
                if isinstance(segmentation, list):
                    segmentation = segmentation[0]
                subgroup.create_dataset("segmentation",
                                        shape=segmentation.shape,
                                        data=segmentation,
                                        dtype=np.int16, **zarr_kwargs)
            
            # Iterate through volume batch-wise.
            for idx0, idx1 in zip(range(0, len(volume[0]), batch_size),
                   range(batch_size, len(volume[0])+batch_size+1, batch_size)):
                # Prepare data for joint evaluation and prediction.
                if evaluate:
                    batch = (volume[0][idx0:idx1], volume[1][idx0:idx1])
                    x, y, sample_weights = model._standardize_user_data(
                                                            batch[0], batch[1])
                    ins = x+y+sample_weights
                else:
                    batch = (volume[0][idx0:idx1],)
                    ins = _standardize_input_data(batch[0],
                                                  model._feed_input_names,
                                                  model._feed_input_shapes,
                                                  check_batch_axis=False,
                                                  exception_prefix='input')
                if model.uses_learning_phase and \
                        not isinstance(K.learning_phase(), int):
                    ins += [0.]
                
                # Jointly evaluate and predict.
                outputs = predict_function(ins)
                if num_outputs==1:
                    predictions = outputs[0:1]
                    if evaluate:
                        val_metrics = outputs[1:]
                elif num_outputs==2:
                    predictions = outputs[0:2]
                    if evaluate:
                        val_metrics = outputs[2:]
                else:
                    raise ValueError("num_outputs must be 1 or 2")
                
                # Write predictions.
                predictions = np.concatenate(predictions, axis=1)
                subgroup['volume'][idx0:idx1] = predictions
                
                # Update metrics
                if evaluate:
                    val_logs = OrderedDict(zip(model.metrics_names,
                                               val_metrics))
                    val_logs.update({'batch': batch_num,
                                     'size': len(batch[0])})
                    val_callbacks.on_batch_end(batch_num, val_logs)
                
                batch_num += 1
            
    if evaluate:
        # Update metrics
        val_callbacks.on_epoch_end(0, val_logs)
    
        # Output metrics
        for m in val_logs:
            if m not in ['batch', 'size']:
                print("{}: {}".format(m, val_logs[m]))


def run(general_settings,
        model_kwargs,
        data_gen_kwargs,
        predict_kwargs,
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
    predict_kwargs.update({'volume_indices': volume_indices})

    '''
    Print settings to screen
    '''
    all_dicts = OrderedDict((
        ("General settings", general_settings),
        ("Model settings", model_kwargs),
        ("Data generator settings", data_gen_kwargs),
        ("Prediction settings", predict_kwargs)
        ))
    if loader_kwargs is not None:
        all_dicts['loader_kwargs'] = loader_kwargs
    print("Experiment:", general_settings['save_subdir'])
    print("")
    for name, d in all_dicts.items():
        print("#### {} ####".format(name))
        for key in d.keys():
            print(key, ":", d[key])
        print("")
        
    '''
    Model type - backward compatibility in argument passing style.
    '''
    if not 'model_type' in model_kwargs:
        if 'two_levels' in model_kwargs:
            if model_kwargs.pop('two_levels')==True:
                model_kwargs['model_type'] = 'two_levels'
        if 'adversarial' in model_kwargs:
            if model_kwargs.pop('adversarial')==True:
                raise ValueError("adversarial model not supported")
        if 'multi_slice' in model_kwargs:
            if model_kwargs.pop('multi_slice')==True:
                model_kwargs['model_type'] = 'multi_slice'
    elif model_kwargs['model_type']=='adversarial':
        raise ValueError("adversarial model not supported")
        
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
                            "    r : delete and replace directory\n"))
        if write_into in ['n', '']:
            print("Aborted")
            sys.exit()
        if write_into=='r':
            print("WARNING: Deleting existing results directory.")
            shutil.rmtree(experiment_dir)
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
            load_and_freeze_weights(model, load_path, **loader_kwargs)
    
    '''
    Run model
    '''
    predict(model=model,
            save_path=experiment_dir,
            data_gen_kwargs=data_gen_kwargs,
            initial_epoch=initial_epoch,
            **predict_kwargs)
