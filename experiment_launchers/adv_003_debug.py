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
    ('results_dir', "/home/eugene/Experiments/lits/results"),
    ('save_subdir', "adv/003_debug"),
    ('load_subpath', None),
    ('random_seed', 1234),
    ('num_train', 100),
    ('layers_to_not_freeze', None),
    ('exclude_data',[32, 34, 38, 41, 47, 83, 87, 89, 91,
                     101, 105, 106, 114, 115, 119]),
    ('freeze', False),
    ('evaluate', False)
    ))

model_kwargs = OrderedDict((
    ('input_shape', (1, 256, 256)),
    ('num_classes', 1),
    ('num_init_blocks', 3),
    ('num_main_blocks', 0),
    ('main_block_depth', 1),
    ('input_num_filters', 8),
    ('num_cycles', 1),
    ('weight_decay', 0.0005), 
    ('dropout', 0.05),
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
    ('init', 'he_normal'),
    ('two_levels', True),
    ('adversarial', True),
    ('ndim', 2)
    ))
    
discriminator_kwargs = OrderedDict((
    ('input_shape', (2, 256, 256)),
    ('num_classes', 1),
    ('num_init_blocks', 3),
    ('num_main_blocks', 0),
    ('main_block_depth', 1),
    ('input_num_filters', 4),
    ('mainblock', basic_block),
    ('initblock', basic_block_mp),
    ('dropout', 0.05),
    ('normalization', BatchNormalization),
    ('weight_decay', 0.0005),
    ('norm_kwargs', {'momentum': 0.9,
                     'scale': True,
                     'center': True,
                     'axis': 1}),
    ('init', 'he_normal'),
    ('output_activation', 'sigmoid'),
    ('ndim', 2)
    ))

data_gen_kwargs = OrderedDict((
    ('data_path', "/store/Data/lits_challenge/sorted/data_liver.zarr"),
    ('nb_io_workers', 1),
    ('nb_proc_workers', 1),
    ('downscale', True),
    ('num_consecutive', None)
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
    ('val_batch_size', 150),
    ('num_epochs', 200),
    ('max_patience', 50),
    
    # optimizer
    ('optimizer', 'RMSprop'),   # 'RMSprop', 'nadam', 'adam', 'sgd'
    ('learning_rate', 0.01),
    
    # other
    ('show_model', False),
    ('save_every', 10),         # Save predictions every x epochs
    ('mask_to_liver', False),
    ('liver_only', False),
    ('adversarial_weight', 1.)
    ))
train_kwargs['num_outputs'] = model_kwargs['num_outputs']

run(general_settings=general_settings,
    model_kwargs=model_kwargs,
    discriminator_kwargs=discriminator_kwargs,
    data_gen_kwargs=data_gen_kwargs,
    data_augmentation_kwargs=data_augmentation_kwargs,
    train_kwargs=train_kwargs)
 
