import sys
sys.path.append("..")
from collections import OrderedDict
from keras.layers import BatchNormalization
from keras.initializers import VarianceScaling
from lib.blocks import (bottleneck,
                        basic_block,
                        basic_block_mp)
from lib.predict import run
import os


general_settings = OrderedDict((
    ('results_dir', "/home/eugene/Experiments/lits/results"),
    ('save_subdir', "3D/eval/006c_f2"),
    ('load_subpath', "3D/006c_f2/best_weights_ldice.hdf5"),
    ('random_seed', 1234),
    ('num_train', 100),
    ('exclude_data',[32, 34, 38, 41, 47, 83, 87, 89, 91,
                     101, 105, 106, 114, 115, 119]),
    ))
    
loader_kwargs = OrderedDict((
    ('freeze', False),
    ('verbose', True),
    ('layers_to_not_freeze', None),
    ('freeze_mask', ["classifier"]),
    ('load_mask', None),
    ('depth_offset', 0)
    ))

model_kwargs = OrderedDict((
    ('model_type', 'multi_slice'),
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
    ('init', 'he_normal'),
    ('ms_ndim_out', 3),
    ('ndim', 3),
    ('nonlinearity', 'relu')
    ))

data_gen_kwargs = OrderedDict((
    ('data_path', "/store/Data/lits_challenge/sorted/data_liver.zarr"),
    ('nb_io_workers', 1),
    ('nb_proc_workers', 0),
    ('downscale', False),
    ('num_consecutive', 1)
    ))

predict_kwargs = OrderedDict((
    # data
    ('batch_size', 50),
    
    # other
    ('show_model', False),
    ('mask_to_liver', False),
    ('liver_only', False),
    ('save_predictions', True),
    ('evaluate', True)
    ))
predict_kwargs['num_outputs'] = model_kwargs['num_outputs']
predict_kwargs['num_classes'] = model_kwargs['num_classes']

run(general_settings=general_settings,
    model_kwargs=model_kwargs,
    data_gen_kwargs=data_gen_kwargs,
    predict_kwargs=predict_kwargs)
 
