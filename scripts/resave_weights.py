# Import python libraries
from collections import OrderedDict
import numpy as np
import os
import sys

# Import keras libraries
# Import in-house libraries
from model.model import assemble_model
from model.blocks import (bottleneck,
                          basic_block,
                          basic_block_mp)


load_path = os.path.join("/home/imagia/eugene.vorontsov-home/",
                         "Experiments/lits/results/",
                         "stage1/029/best_weights.hdf5")
save_path = os.path.join("/home/imagia/eugene.vorontsov-home/",
                         "Experiments/lits/results/",
                         "stage1/029/best_weights_named.hdf5")

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
    ('num_residuals', 1),
    ('num_first_conv', 1),
    ('num_final_conv', 1),
    ('num_classifier', 1),
    ('init', 'he_normal')
    ))
    
    
'''
Assemble model
'''
# Increase the recursion limit to handle resnet skip connections
print("Assembling model")
sys.setrecursionlimit(99999)
model = assemble_model(**model_kwargs)
print("   number of parameters : ", model.count_params())


'''
Load and re-save weights
'''
print("Loading and re-saving weights")
model.load_weights(load_path)
model.save_weights(save_path)
