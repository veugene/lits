merge_dirs = ["/home/eugene/Experiments/lits/results/3D/008b_f2__eval/"
              "output_volumes_pred_liver_3_ldice/test/"]
#save_merged_to = "/home/eugene/Experiments/lits/results/_FINAL/merged/008b_f2"
save_final_to = "/home/eugene/Experiments/lits/results/_FINAL/" \
                "submbit/008b_f2"
            
START = 0
STOP = 130

import os
import re
import SimpleITK as sitk
import numpy as np
from scipy import ndimage
from data_tools.binary_morphology import binary_dilation

if len(merge_dirs)>1:
    if not os.path.exists(save_merged_to):
        os.makedirs(save_merged_to)

    for fn in sorted(os.listdir(merge_dirs[0]))[START:STOP]:
        print("Merging {}".format(fn))
        volumes = []
        for md in merge_dirs:
            f = sitk.ReadImage(os.path.join(md, fn))
            volumes.append(sitk.GetArrayFromImage(f))
        merged_volume = np.mean(volumes, axis=0)
        f_merged = sitk.GetImageFromArray(merged_volume)
        f_merged.CopyInformation(f)
        sitk.WriteImage(f_merged, os.path.join(save_merged_to, fn))
else:
    save_merged_to = merge_dirs[0]


def largest_connected_component(mask):
    labels, num_components = ndimage.label(mask)
    if not num_components:
        raise ValueError("The mask is empty.")
    if num_components==1:
        return mask.astype(np.bool)
    label_count = np.bincount(labels.ravel().astype(np.int))
    label_count[0] = 0      # discard 0 the 0 label
    return labels == label_count.argmax()


def select_slices(liver_volume, liver_extent):
    """
    Find the largest connected component in the liver prediction volume and 
    determine which axial slices include this component. Return the indices
    for these slices as well as of `liver_extent` slices adjacent to the liver,
    above and below.
    
    All nonzero values in the liver_volume are taken as liver labels.
    """
    mask = largest_connected_component(liver_volume>=0.5)
    indices = np.unique(np.where(mask)[0])
    if liver_extent:
        min_idx = max(0, indices[0]-liver_extent)
        max_idx = min(len(mask), indices[-1]+liver_extent+1)
        indices = [i for i in range(min_idx, max_idx)]
    return indices, mask

if not os.path.exists(save_final_to):
    os.makedirs(save_final_to)

for fn in sorted(os.listdir(save_merged_to))[START:STOP]:
    if fn.endswith("_output_0.nii.gz"):
        print("Finalizing {}".format(fn))
        prediction_f = sitk.ReadImage(os.path.join(save_merged_to, fn))
        prediction_np = sitk.GetArrayFromImage(prediction_f)
        
        # Load liver
        liver_f = sitk.ReadImage(os.path.join(save_merged_to,
                              fn[:-len("_output_0.nii.gz")]+"_output_1.nii.gz"))
        liver_np = sitk.GetArrayFromImage(liver_f)
        
        # Crop and select largest component as liver
        indices, liver_np = select_slices(liver_np, 3)
        prediction_np[:indices[0]] = 0
        prediction_np[indices[-1]:] = 0
        
        # Dilate liver and crop to dilated liver
        print("DEBUG spacing: ", liver_f.GetSpacing()[::-1], liver_np.shape)
        dilated_liver = binary_dilation(liver_np,
                                        spacing=liver_f.GetSpacing()[::-1],
                                        radius=10,
                                        nb_workers=4)
        prediction_np[dilated_liver==0] = 0
        
        # Save as segmetation
        segmentation = np.zeros(prediction_np.shape, np.int16)
        segmentation[liver_np>0] = 1
        segmentation[prediction_np>=0.5] = 2
        segmentation_f = sitk.GetImageFromArray(segmentation)
        segmentation_f.CopyInformation(prediction_f)
        fn_num = re.search(r'\d+', fn).group()
        segmentation_fn = "test-segmentation-"+fn_num+".nii"
        print("Writing to {}".format(segmentation_fn))
        sitk.WriteImage(segmentation_f, os.path.join(save_final_to,
                                                     segmentation_fn))
        sitk.WriteImage(segmentation_f, os.path.join(save_final_to,
                                                     segmentation_fn+".gz"))
