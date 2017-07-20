import sys
import os
sys.path.append("..")
from collections import OrderedDict
from keras.layers import BatchNormalization

from lib.blocks import (bottleneck,
                        basic_block,
                        basic_block_mp)
from lib.train import run
from lib.normalization_layers import LayerNorm


general_settings = OrderedDict((
    ('results_dir', "/home/eugene/Experiments/lits/results"),
    ('save_subdir', "lstm/001"),
    ('load_subpath', None),
    ('random_seed', 1234),
    ('num_train', 100),
    ('exclude_data',[32, 34, 38, 41, 47, 83, 87, 89, 91,
                     101, 105, 106, 114, 115, 119]),
    ('evaluate', False)
    ))
    
loader_kwargs = OrderedDict((
    ('freeze', False),
    ('verbose', True),
    ('layers_to_not_freeze', None),
    ('freeze_mask', None),
    ('load_mask', None),
    ('depth_offset', 0)
    ))

model_kwargs = OrderedDict((
    ('model_type', 'lstm'),
    ('input_shape', (5, 3, 64, 512, 512)),
    ('num_filters', 32),
    ('weight_decay', 0.0005), 
    ('weight_norm', False),
    ('normalization', LayerNorm),
    ('norm_kwargs', {'scale': True,
                     'center': True}),
    ('init', 'he_normal'),
    ('num_outputs', 1),
    ('num_classes', 1)
    ))

data_gen_kwargs = OrderedDict((
    ('data_path', "/home/eugene/Experiments/lits/results/orig_rerun/"
                  "001-2_predict/predictions.zarr"),
    ('nb_io_workers', 1),
    ('nb_proc_workers', 0),
    ('downscale', False),
    ('num_consecutive', None),
    ('recurrent', True),
    ('truncate_every', 3)
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
    ('batch_size', 5),
    ('val_batch_size', 5),
    ('num_epochs', 200),
    ('max_patience', 50),
    
    # optimizer
    ('optimizer', 'RMSprop'),   # 'RMSprop', 'nadam', 'adam', 'sgd'
    ('learning_rate', 0.001),
    
    # other
    ('show_model', False),
    ('save_every', 1),         # Save predictions every x epochs
    ('mask_to_liver', False),
    ('liver_only', False)
    ))
train_kwargs['num_outputs'] = model_kwargs['num_outputs']

run(general_settings=general_settings,
    model_kwargs=model_kwargs,
    data_gen_kwargs=data_gen_kwargs,
    data_augmentation_kwargs=data_augmentation_kwargs,
    train_kwargs=train_kwargs,
    loader_kwargs=loader_kwargs)
