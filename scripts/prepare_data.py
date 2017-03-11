import numpy as np
import SimpleITK as sitk
import os
import zarr
import random
import scipy.misc


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
        if not hasattr(exclude_class, '__len__'):
            exclude_class = [exclude_class]
        for c in exclude_class:
            exclude_indices = np.where(segm==c)
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
def create_dataset(save_name, save_dir, data_dir, proportions):
    save_name += ".zarr"
    save_path = os.path.join(save_dir, save_name)

    zgroup = zarr.open_group(store=save_path, mode='w', path="/")
    zarr_kwargs = {'chunks': (1, 512, 512),
                   'compressor': zarr.Blosc(cname='lz4', clevel=9, shuffle=1)}

    for i in range(130):
        print("Processing volume {}".format(i))
        volume = sitk.ReadImage(os.path.join(data_dir,
                                             "volume-"+str(i)+".nii"))
        volume_np = sitk.GetArrayFromImage(volume)
        seg = sitk.ReadImage(os.path.join(data_dir,
                                          "segmentation-"+str(i)+".nii"))
        seg_np = sitk.GetArrayFromImage(seg)

        slices = []
        if proportions[0] > 0:
            slices.extend(get_slices(seg_np, target_class=0,
                                     exclude_class=[1, 2],
                                     proportion=proportions[0]))
        if proportions[1] > 0:
            slices.extend(get_slices(seg_np, target_class=1,
                                     exclude_class=2,
                                     proportion=proportions[1]))
        if proportions[2] > 0:
            slices.extend(get_slices(seg_np, target_class=2,
                                     proportion=proportions[2]))

        volume_np = volume_np[slices]
        seg_np = seg_np[slices]

        # Sanity check
        # save_slices(seg_np)
        
        if len(volume_np)==0:
            print("WARNING! Skipping empty volume #{}".format(i))
            continue
        
        print("Saving {} slices".format(volume_np.shape[0]))
        subgroup = zgroup.create_group(str(i))
        subgroup.create_dataset("volume", shape=volume_np.shape,
                                data=volume_np,
                                dtype=np.float32, **zarr_kwargs)
        subgroup.create_dataset("segmentation", shape=seg_np.shape,
                                data=seg_np,
                                dtype=np.int16, **zarr_kwargs)


if __name__=='__main__':
    # Define load and save directories
    data_dir = "/export/projects/Candela/datasets/lits_challenge/all/"
    save_dir = "/data/TransientData/Candela/lits_challenge/"
    
    ## Save lesion dataset
    #print("########## Preparing lesion dataset ##########")
    #proportions = {0: 0., 1: 0., 2: 1.}
    #create_dataset("data_lesions", save_dir=save_dir, data_dir=data_dir,
                   #proportions=proportions)
    
    ## Save liver dataset
    #print("########## Preparing liver dataset ##########")
    #proportions = {0: 0., 1: 1., 2: 1.}
    #create_dataset("data_liver", save_dir=save_dir, data_dir=data_dir,
                   #proportions=proportions)
    
    # Save complete dataset
    print("########## Preparing complete dataset ##########")
    proportions = {0: 1., 1: 1., 2: 1.}
    create_dataset("data_all", save_dir=save_dir, data_dir=data_dir,
                   proportions=proportions)
