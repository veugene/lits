import numpy as np
import SimpleITK as sitk
import os
import zarr
import random
import scipy.misc

data_dir = "/export/projects/Candela/datasets/lits_challenge/all/"
save_dir = "/export/projects/Candela/datasets/by_project/lits/"
save_name = "data_extended"

# Define class inclusion percentage
proportion_liver = 0.5
proportion_bg = 0.3

def get_slices(segm, target_class, exclude_class=None, proportion=1.0):
    """
    Return slices corresponding to a target segmentation class
    
    :param segm: Segmentation mask in numpy.array
    :param target_class: Only slices containing this class are returned
    :param exclude_class: Only slices excluding this class are returned
    :param proportion: Percentage of available slices returned randomly
    """

    indices = np.where(segm==target_class)
    slices = np.unique(indices[0])

    if exclude_class is not None:
        exclude_indices = np.where(segm==exclude_class)
        exclude_slices = np.unique(exclude_indices[0])
        slices = [x for x in slices if x not in exclude_slices]
        
    if proportion < 1.0:
        M = list(range(len(slices)))
        random.shuffle(M)
        n = proportion * len(slices)
        n = round(n)
        slices = np.array(slices)
        slices = slices[M[:n]]
 
    return slices

def save_slices(vol_np):
    """
    Save all slices of a given volume to JPEG in tmp/
    
    :param vol_np: volume to be saved 
    """
    for s in range(len(vol_np)):
        scipy.misc.imsave('/tmp/test'+str(s)+'.jpg', vol_np[s])

"""
Starting the data preparation script
"""
save_name += ".zarr"
save_path = os.path.join(save_dir, save_name)

zgroup = zarr.open_group(store=save_path, mode='w', path="/")
zarr_kwargs = {'chunks': (1, 512, 512),
               'compressor': zarr.Blosc(cname='lz4', clevel=9, shuffle=1)}

for i in range(130):
    print("Processing volume {}".format(i))
    volume = sitk.ReadImage(os.path.join(data_dir, "volume-"+str(i)+".nii"))
    volume_np = sitk.GetArrayFromImage(volume)
    seg = sitk.ReadImage(os.path.join(data_dir, "segmentation-"+str(i)+".nii"))
    seg_np = sitk.GetArrayFromImage(seg)

    slices = []
    if proportion_bg > 0:
        slices.extend(get_slices(seg_np, target_class=0, exclude_class=1, proportion=proportion_bg))
    if proportion_liver > 0:
        slices.extend(get_slices(seg_np, target_class=1, exclude_class=2, proportion=proportion_liver))
    slices.extend(get_slices(seg_np, target_class=2))

    volume_np = volume_np[slices]
    seg_np = seg_np[slices]

    # Sanity check
    # save_slices(seg_np)
    
    if len(volume_np)==0:
        print("WARNING! Skipping empty volume #{}".format(i))
        continue
    
    print("Saving {} slices".format(volume_np.shape[0]))
    subgroup = zgroup.create_group(str(i))
    subgroup.create_dataset("volume", shape=volume_np.shape, data=volume_np,
                            dtype=np.float32, **zarr_kwargs)
    subgroup.create_dataset("segmentation", shape=seg_np.shape, data=seg_np,
                            dtype=np.int16, **zarr_kwargs)
