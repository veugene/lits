from collections import OrderedDict
from lib.blocks import (bottleneck,
                        basic_block,
                        basic_block_mp)
from lib.train import run
import os

first_conv = ['first_conv_0']
final_conv = ['final_conv_0']
final_bn = ['final_bn_0']

branch0_bn = ['a_basic_block_1__1_bn',
              'a_basic_block_1__2_bn',
              'd1_basic_block_mp_1_bn_0',
              'd2_basic_block_mp_1_bn_0',
              'd3_basic_block_1__1_bn',
              'd3_basic_block_1__2_bn',
              'd4_basic_block_1__1_bn',
              'd4_basic_block_1__2_bn',
              'd5_basic_block_1__1_bn',
              'd5_basic_block_1__2_bn',
              'u1_basic_block_mp_1_bn_0',
              'u2_basic_block_mp_1_bn_0',
              'u3_basic_block_1__1_bn',
              'u3_basic_block_1__2_bn',
              'u4_basic_block_1__1_bn',
              'u4_basic_block_1__2_bn',
              'u5_basic_block_1__1_bn',
              'u5_basic_block_1__2_bn']

branch0_other = ['a_basic_block_1__1_conv2d',
                 'a_basic_block_1__2_conv2d',
                 'd1_basic_block_mp_1_conv2d_0',
                 'd2_basic_block_mp_1_conv2d_0',
                 'd3_basic_block_1__1_conv2d',
                 'd3_basic_block_1__2_conv2d',
                 'd4_basic_block_1__1_conv2d',
                 'd4_basic_block_1__2_conv2d',
                 'd5_basic_block_1__1_conv2d',
                 'd5_basic_block_1__2_conv2d',
                 'u1_basic_block_mp_1_conv2d_0',
                 'u2_basic_block_mp_1_conv2d_0',
                 'u3_basic_block_1__1_conv2d',
                 'u3_basic_block_1__2_conv2d',
                 'u4_basic_block_1__1_conv2d',
                 'u4_basic_block_1__2_conv2d',
                 'u5_basic_block_1__1_conv2d',
                 'u5_basic_block_1__2_conv2d']

branch1_bn = ['a_basic_block_1__3_bn',
              'a_basic_block_1__4_bn',
              'd1_basic_block_mp_1_bn_1',
              'd2_basic_block_mp_1_bn_1',
              'd3_basic_block_1__3_bn',
              'd3_basic_block_1__4_bn',
              'd4_basic_block_1__3_bn',
              'd4_basic_block_1__4_bn',
              'd5_basic_block_1__3_bn',
              'd5_basic_block_1__4_bn',
              'u1_basic_block_mp_1_bn_1',
              'u2_basic_block_mp_1_bn_1',
              'u3_basic_block_1__3_bn',
              'u3_basic_block_1__4_bn',
              'u4_basic_block_1__3_bn',
              'u4_basic_block_1__4_bn',
              'u5_basic_block_1__3_bn',
              'u5_basic_block_1__4_bn']

branch1_other = ['a_basic_block_1__3_conv2d',
                 'a_basic_block_1__4_conv2d',
                 'd1_basic_block_mp_1_conv2d_1',
                 'd2_basic_block_mp_1_conv2d_1',
                 'd3_basic_block_1__3_conv2d',
                 'd3_basic_block_1__4_conv2d',
                 'd4_basic_block_1__3_conv2d',
                 'd4_basic_block_1__4_conv2d',
                 'd5_basic_block_1__3_conv2d',
                 'd5_basic_block_1__4_conv2d',
                 'u1_basic_block_mp_1_conv2d_1',
                 'u2_basic_block_mp_1_conv2d_1',
                 'u3_basic_block_1__3_conv2d',
                 'u3_basic_block_1__4_conv2d',
                 'u4_basic_block_1__3_conv2d',
                 'u4_basic_block_1__4_conv2d',
                 'u5_basic_block_1__3_conv2d',
                 'u5_basic_block_1__4_conv2d']

classifiers = ['classifier_conv_0',
               'classifier_conv_1']

skips = ['d4_basic_block_1_shortcut_1_conv2d',
                    'd5_basic_block_1_shortcut_1_conv2d',
                    'long_skip_up_3',
                    'long_skip_up_4']

general_settings = OrderedDict((
    ('results_dir', os.path.join("/home/imagia/eugene.vorontsov-home/",
                                    "Experiments/lits/results")),
    ('save_subdir', "stage2/031f_05_l1"),
    ('load_subpath', "stage2/031f_05/best_weights_dice.hdf5"),
    ('random_seed', 1234),
    ('num_train', 100),
    ('layers_to_not_freeze', final_bn+branch0_bn+branch0_other+branch1_bn\
                             +[classifiers[0]]),
    ('exclude_data',[32, 34, 38, 41, 47, 83, 87, 89, 91,
                     101, 105, 106, 114, 115, 119]),
    ('freeze', True)
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
    ('num_outputs', 1),
    ('init', 'zero')
    ))

data_gen_kwargs = OrderedDict((
    ('data_path', os.path.join("/data/TransientData/Candela/",
                                "lits_challenge/data_lesions.zarr")),
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
    ('save_every', 10),         # Save predictions every x epochs
    ))

run(general_settings=general_settings,
    model_kwargs=model_kwargs,
    data_gen_kwargs=data_gen_kwargs,
    data_augmentation_kwargs=data_augmentation_kwargs,
    train_kwargs=train_kwargs)
