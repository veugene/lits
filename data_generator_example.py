import numpy as np
from collections import OrderedDict

from lib.utils import data_generator


"""
Identify indices of volumes to use for the training set
and for the validation set.

NOTE: the validation indices listed below are those used for LITS.
"""
volume_indices = {}
volume_indices['valid'] = [118, 117, 48, 30, 26, 120, 23,
                           54, 15, 24, 81, 58, 42, 92, 52]
volume_indices['train'] = [i for i in np.arange(130) \
                                           if i not in volume_indices['valid']]

"""
Set some data augmentation arguments that work well for LITS.
"""
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

"""
Set some generic data generator arguments:

 data_path : location of zarr file for data
 nb_io_workers : number of parallel threads for loading data
 nb_proc_workers : number of parallel processes for preprocessing data
 downscale : whether to downscale slices (volumes have all axial slices 
    downscaled). Downscaling is 2x (i.e. 512x512 becomes 256x256).
 num_consecutive : (see description of data_generator in utils.py)
 
NOTE: The data_path below exists on 10.10.6.106 (and some other systems).

The data_liver.zarr contains, for each volume, only the slices that contain
the liver.
"""
data_gen_kwargs = OrderedDict((
    ('data_path', "/data/TransientData/Candela/lits_challenge/data_liver.zarr"),
    ('nb_io_workers', 1),
    ('nb_proc_workers', 4),
    ('downscale', True),
    ('num_consecutive', None)
    ))


"""
Set up the generators. See lib/train.py for a real example.
"""
gen = {}
gen['train'] = data_generator(volume_indices=volume_indices['train'],
                              mode='volumes',           # return volumes
                              batch_size=2,
                              shuffle=True,
                              loop_forever=True,        # keras likes this
                              transform_kwargs=data_augmentation_kwargs,
                              **data_gen_kwargs)
gen['valid'] = data_generator(volume_indices=volume_indices['valid'],
                              mode='volumes',
                              batch_size=4,
                              shuffle=False,
                              loop_forever=True,
                              transform_kwargs=None,
                              **data_gen_kwargs)

"""
Train a model.

NOTE that the generators objects have a __len__ attribute (returns number of
batches) and an actual data generator is returned from their flow() method.
"""

# Just for this example, print the arguments.
class Model(object):
    def fit_generator(self, **kwargs):
        print("fit_generator arguments are: ")
        for key, val in kwargs.items():
            print("{} : {}".format(key, val))
model = Model()

# The example call for a model.
model.fit_generator(generator=gen['train'].flow(),
                    steps_per_epoch=len(gen['train']),
                    epochs=100,
                    validation_data=gen['valid'].flow(),
                    validation_steps=len(gen['valid']))

#"""
#Test data flow.
#"""
#print("")
#for key in ['train', 'valid']:
    #print("Testing {} flow".format(key))
    #flow = gen[key].flow()
    #for i in range(5):    
        #batch = next(flow)
        #print("batch_num: {}, shapes_0: {}, shapes_1: {}"
              #"".format(i,
                        #[np.shape(b) for b in batch[0]],
                        #[np.shape(b) for b in batch[1]]))
