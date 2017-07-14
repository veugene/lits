#merge_dirs = ["/home/eugene/Experiments/lits/results/test_data/101_01-1__eval/"
              #"output_volumes_ldice_4views",
              #"/home/eugene/Experiments/lits/results/test_data/101_02-1__eval/"
              #"output_volumes_ldice_4views",
              #"/home/eugene/Experiments/lits/results/test_data/102-1__eval/"
              #"output_volumes_ldice_4views",
              #"/home/eugene/Experiments/lits/results/test_data/101_01-1__eval/"
              #"output_volumes_dice_4views",
              #"/home/eugene/Experiments/lits/results/test_data/101_02-1__eval/"
              #"output_volumes_dice_4views",
              #"/home/eugene/Experiments/lits/results/test_data/102-1__eval/"
              #"output_volumes_dice_4views"]

#merge_dirs = ["/home/eugene/Experiments/lits/results/test_data/102-1__eval/"
              #"output_volumes_ldice_4views",
              #"/home/eugene/Experiments/lits/results/test_data/101_02-1__eval/"
              #"output_volumes_dice_4views",
              #"/home/eugene/Experiments/lits/results/test_data/102-1__eval/"
              #"output_volumes_dice_4views"]

#save_to = "/home/eugene/Experiments/lits/results/test_data/" \
          #"submission2/merged_4views_dice_ldice"
#save_final_to = "/home/eugene/Experiments/lits/results/test_data/" \
                #"submission2/merged_4views_dice_ldice_final"
            
merge_dirs = ["/home/eugene/Experiments/lits/results/validation/as_sub2/"
              "102-1__eval/output_volumes_pred_liver_3_ldice_4views",
              "/home/eugene/Experiments/lits/results/validation/as_sub2/"
              "101_02-1__eval/output_volumes_pred_liver_3_dice_4views",
              "/home/eugene/Experiments/lits/results/validation/as_sub2/"
              "102-1__eval/output_volumes_pred_liver_3_dice_4views"]

save_to = "/home/eugene/Experiments/lits/results/validation/" \
          "as_sub2/merged_4views_dice_ldice"
save_final_to = "/home/eugene/Experiments/lits/results/validation/" \
                "as_sub2/merged_4views_dice_ldice_final"
            
START = 0
STOP = 130

import os
import SimpleITK as sitk
import numpy as np
from scipy import ndimage
from data_tools.binary_morphology import binary_dilation

if not os.path.exists(save_to):
    os.makedirs(save_to)

for fn in sorted(os.listdir(merge_dirs[0]))[START:STOP]:
    print("Merging {}".format(fn))
    volumes = []
    for md in merge_dirs:
        f = sitk.ReadImage(os.path.join(md, fn))
        volumes.append(sitk.GetArrayFromImage(f))
    merged_volume = np.mean(volumes, axis=0)
    f_merged = sitk.GetImageFromArray(merged_volume)
    f_merged.CopyInformation(f)
    sitk.WriteImage(f_merged, os.path.join(save_to, fn))


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
#a = sorted(os.listdir(save_to))
#for i in range(len(a)):
    #print(i, a[i])

for fn in sorted(os.listdir(save_to))[START:STOP]:
    if fn.endswith("_output_0.nii.gz"):
        print("Finalizing {}".format(fn))
        prediction_f = sitk.ReadImage(os.path.join(save_to, fn))
        prediction_np = sitk.GetArrayFromImage(prediction_f)
        
        # Load liver
        liver_f = sitk.ReadImage(os.path.join(save_to,
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
                                        radius=20,
                                        nb_workers=4)
        prediction_np[dilated_liver==0] = 0
        
        # Save as segmetation
        segmentation = np.zeros(prediction_np.shape, np.int16)
        segmentation[liver_np>0] = 1
        segmentation[prediction_np>=0.5] = 2
        segmentation_f = sitk.GetImageFromArray(segmentation)
        segmentation_f.CopyInformation(prediction_f)
        fn_num = fn.split('-')[1].split('_')[0]
        segmentation_fn = "test-segmentation-"+fn_num+".nii"
        print("Writing to {}".format(segmentation_fn))
        sitk.WriteImage(segmentation_f, os.path.join(save_final_to,
                                                     segmentation_fn))
        sitk.WriteImage(segmentation_f, os.path.join(save_final_to,
                                                     segmentation_fn+".gz"))
