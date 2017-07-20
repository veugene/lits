import sys
sys.path.append("..")
from collections import OrderedDict
from keras.layers import BatchNormalization
from keras.initializers import VarianceScaling
from lib.blocks import (bottleneck,
                        basic_block,
                        basic_block_mp)
from lib.train import run
import os


general_settings = OrderedDict((
    ('results_dir', "/export/home/eugene/Experiments/lits/results"),
    ('save_subdir', "3D/008b_f2"),
    ('load_subpath', "3D/008b/best_weights_ldice.hdf5"),
    ('random_seed', 1234),
    ('num_train', 100),
    ('exclude_data',[32, 34, 38, 41, 47, 83, 87, 89, 91,
                     101, 105, 106, 114, 115, 119]),
    ('evaluate', False)
    ))
    
loader_kwargs = OrderedDict((
    ('freeze', True),
    ('verbose', True),
    ('layers_to_not_freeze', None),
    ('freeze_mask', ['classifier', 'conv_3D']),
    ('freeze_mask', None),
    ('load_mask', None),
    ('depth_offset', 0)
    ))

model_kwargs = OrderedDict((
    ('input_shape', (1, 3, 512, 512)),
    ('num_classes', 1),
    ('num_init_blocks', 2),
    ('num_main_blocks', 3),
    ('main_block_depth', 1),
    ('input_num_filters', 32),
    ('num_cycles', 1),
    ('weight_decay', 0.0005), 
    ('dropout', 0.05),
    ('weight_norm', False),
    ('normalization', BatchNormalization),
    ('mainblock', basic_block),
    ('initblock', basic_block_mp),
    ('norm_kwargs', {'momentum': 0.9,
                     'scale': True,
                     'center': True,
                     'axis': 1}),
    ('cycles_share_weights', True),
    ('num_residuals', 1),
    ('num_first_conv', 1),
    ('num_final_conv', 1),
    ('num_classifier', 1),
    ('num_outputs', 2),
    ('init', VarianceScaling(scale=1., mode='fan_in', distribution='normal')),
    ('nonlinearity', 'selu'),
    ('two_levels', True),
    ('multi_slice', True),
    ('ndim', 3)
    ))

data_gen_kwargs = OrderedDict((
    ('data_path', "/data/TransientData/Candela/lits_challenge/"
                  "data_liver.zarr"),
    ('nb_io_workers', 1),
    ('nb_proc_workers', 4),
    ('downscale', False),
    ('num_consecutive', 1)
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
    ('batch_size', 3),
    ('val_batch_size', 15),
    ('num_epochs', 200),
    ('max_patience', 50),
    
    # optimizer
    ('optimizer', 'RMSprop'),   # 'RMSprop', 'nadam', 'adam', 'sgd'
    ('learning_rate', 0.0001),
    
    # other
    ('show_model', False),
    ('save_every', 0),         # Save predictions every x epochs
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
