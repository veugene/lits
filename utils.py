import zarr
from scipy import misc as scipy_misc
import numpy as np
from data_tools.data_augmentation import random_transform
from data_tools.io import data_flow
from data_tools.wrap import multi_source_array

def resize_stack(arr, size, interp='bilinear'):
    out = np.zeros((len(arr),)+tuple(size), arr.dtype)
    for i, arr_slice in enumerate(arr):
        out[i] = scipy_misc.imresize(arr_slice, size, interp)
    return out

def data_generator(data_path, volume_indices, batch_size,
                   nb_io_workers=1, nb_proc_workers=0,
                   shuffle=False, loop_forever=False, downscale=False,
                   transform_kwargs=None, data_flow_kwargs=None):
    try:
        zgroup = zarr.open_group(data_path, mode='r')
    except:
        print("Failed to open data: {}".format(data_path))
        raise
    
    # Assemble volumes and corresponding segmentations
    volumes = []
    segmentations = []
    for idx in volume_indices:
        subgroup = zgroup[str(idx)]
        volumes.append(subgroup['volume'])
        segmentations.append(subgroup['segmentation'])
    msa_vol = multi_source_array(source_list=volumes, shuffle=shuffle)
    msa_seg = multi_source_array(source_list=segmentations, shuffle=shuffle)
    
    # Function to rescale the data and do data augmentation, if requested
    def preprocessor(batch):
        b0, b1 = batch
        if downscale:
            b0 = resize_stack(b0, size=(256, 256), interp='bilinear')
            b1 = resize_stack(b1, size=(256, 256), interp='nearest')
        if transform_kwargs is not None:
            b0, b1 = random_transform(b0, b1, **transform_kwargs)
        # standardize
        b0 /= 255.0
        b0 = np.clip(b0, -2.0, 2.0)
        return (np.expand_dims(b0, 1), np.expand_dims(b1, 1))
    
    # Prepare the data iterator
    if data_flow_kwargs is None:
        data_flow_kwargs = {}
    data_gen = data_flow(data=[msa_vol, msa_seg],
                         batch_size=batch_size,
                         nb_io_workers=nb_io_workers,
                         nb_proc_workers=nb_proc_workers,
                         shuffle=shuffle, 
                         loop_forever=loop_forever,
                         preprocessor=preprocessor)
    
    return data_gen
