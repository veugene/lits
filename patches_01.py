from collections import OrderedDict
import os
from lib.patch_training import run

general_settings = OrderedDict((
    ('results_dir', "/home/imagia/eugene.vorontsov-home/"
                    "Experiments/lits/results"),
    ('save_subdir', "patches/debug"),
    ('load_subpath', "patches/01/best_weights_ldice.hdf5"),
    ('random_seed', 1234),
    ('num_train', 100),
    ('exclude_data',[32, 34, 38, 41, 47, 83, 87, 89, 91,
                     101, 105, 106, 114, 115, 119]),
    ))

model_kwargs = OrderedDict((
    ('input_shape', (1, 24, 24)),
    ('num_filters', 32),
    ('num_conv_layers', 2),
    ('dense_layer_width', 300),
    ('weight_decay', 0.0005), 
    ('dropout', 0.05),
    ('batch_norm', True),
    ('bn_kwargs', {'momentum': 0.9, 'mode': 0}),
    ('init', 'he_normal'),
    ))

data_gen_kwargs = OrderedDict((
    #('data_dir', "/data/TransientData/Candela/lits_challenge/patches_24_all/"),
    ('data_dir', "/tmp/patches_24_all/"),
    ('nb_io_workers', 1),
    ('nb_proc_workers', 12),
    ))

#data_augmentation_kwargs = OrderedDict((
    #('rotation_range', 15),
    #('width_shift_range', 0.1),
    #('height_shift_range', 0.1),
    #('shear_range', 0.),
    #('zoom_range', 0.1),
    #('channel_shift_range', 0.),
    #('fill_mode', 'constant'),
    #('cval', 0.),
    #('cvalMask', 0),
    #('horizontal_flip', True),
    #('vertical_flip', True),
    #('rescale', None),
    #('spline_warp', True),
    #('warp_sigma', 0.1),
    #('warp_grid_size', 3),
    #('crop_size', None)
    #))
    
data_augmentation_kwargs = OrderedDict((
    ('rotation_range', 0),
    ('width_shift_range', 0.),
    ('height_shift_range', 0.),
    ('shear_range', 0.),
    ('zoom_range', 0.),
    ('channel_shift_range', 0.),
    ('fill_mode', 'constant'),
    ('cval', 0.),
    ('cvalMask', 0),
    ('horizontal_flip', True),
    ('vertical_flip', True),
    ('rescale', None),
    ('spline_warp', False),
    ('warp_sigma', 0.1),
    ('warp_grid_size', 3),
    ('crop_size', None)
    ))

train_kwargs = OrderedDict((
    # data
    ('batch_size', 400),
    ('val_batch_size', 10000),
    ('num_epochs', 500),
    ('max_patience', 50),
    
    # optimizer
    ('optimizer', 'RMSprop'),   # 'RMSprop', 'nadam', 'adam', 'sgd'
    ('learning_rate', 0.001),    
    ))


run(general_settings=general_settings,
    model_kwargs=model_kwargs,
    data_gen_kwargs=data_gen_kwargs,
    data_augmentation_kwargs=data_augmentation_kwargs,
    train_kwargs=train_kwargs)
