import os
import shutil
import h5py
import zarr
import numpy as np
from data_tools.io import zarr_array_writer


"""
Combine all patches into one array, preshuffled.
"""

patches_dir = "/data/TransientData/Candela/lits_challenge/patches_24_all/"
patch_size = 24
volume_indices = {'train': [68, 90, 46, 6, 77, 57, 8, 96, 113, 124, 116, 97,
                            84, 64, 110, 27, 53, 126, 1, 29, 44, 9, 56, 33, 67,
                            37, 35, 17, 75, 111, 71, 103, 69, 66, 95, 20, 121, 
                            82, 25, 43, 62, 4, 107, 10, 59, 123, 60, 5, 128,
                            39, 7, 98, 73, 100, 99, 79, 21, 18, 122, 104, 88,
                            112, 49, 22, 85, 61, 45, 16, 50, 31, 109, 93, 13,
                            28, 51, 65, 76, 14, 94, 70, 12, 19, 2, 3, 108, 80,
                            0, 11, 72, 127, 36, 40, 125, 55, 129, 78, 86, 74,
                            102, 63],
                  'valid': [118, 117, 48, 30, 26, 120, 23, 54, 15, 24, 81, 58,
                            42, 92, 52]}


if __name__=='__main__':
    writer_kwargs = {'data_element_shape': (1, patch_size, patch_size),
                     'dtype': np.float32,
                     'batch_size': 32,
                     'append': False}
    for f_key in ['train', 'valid']:
        f_path = os.path.join(patches_dir, "patch_set_{}.zarr".format(f_key))
        f_writer = {}
        for arr_key in ['class_1', 'class_2']:
            f_writer[arr_key] = zarr_array_writer(filename=f_path,
                                                  array_name=arr_key,
                                                  **writer_kwargs)
        for idx in volume_indices[f_key]:
            print("Processing `{}` volume {}".format(f_key, idx))
            load_path = os.path.join(patches_dir,
                                     "patch_set_{}.hdf5".format(idx))
            f = h5py.File(load_path, 'r')
            for arr_key in ['class_1', 'class_2']:
                f_writer[arr_key].buffered_write(f[arr_key])
            
