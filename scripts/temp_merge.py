#merge_dirs = ["/home/eugene/Experiments/lits/results/stage3/101_01-1__eval/"
              #"output_volumes_pred_liver_3_ldice",
              #"/home/eugene/Experiments/lits/results/stage3/101_02-1__eval/"
              #"output_volumes_pred_liver_3_ldice",
              #"/home/eugene/Experiments/lits/results/stage3/102-1__eval/"
              #"output_volumes_pred_liver_3_ldice",
              #"/home/eugene/Experiments/lits/results/stage3/101_01-1__eval/"
              #"output_volumes_pred_liver_3_dice",
              #"/home/eugene/Experiments/lits/results/stage3/101_02-1__eval/"
              #"output_volumes_pred_liver_3_dice",
              #"/home/eugene/Experiments/lits/results/stage3/102-1__eval/"
              #"output_volumes_pred_liver_3_dice"]
              
merge_dirs = ["/home/eugene/Experiments/lits/results/stage3/102-1__eval/"
              "output_volumes_pred_liver_3_ldice_4views",
              "/home/eugene/Experiments/lits/results/stage3/101_02-1__eval/"
              "output_volumes_pred_liver_3_dice_4views",
              "/home/eugene/Experiments/lits/results/stage3/102-1__eval/"
              "output_volumes_pred_liver_3_dice_4views"]

save_to = "/home/eugene/Experiments/lits/results/stage3/" \
          "merged_4views_dice_ldice_special"

import os
import SimpleITK as sitk
import numpy as np

if not os.path.exists(save_to):
    os.makedirs(save_to)

for fn in os.listdir(merge_dirs[0]):
    print(fn)
    volumes = []
    for md in merge_dirs:
        f = sitk.ReadImage(os.path.join(md, fn))
        volumes.append(sitk.GetArrayFromImage(f))
    merged_volume = np.mean(volumes, axis=0)
    f_merged = sitk.GetImageFromArray(merged_volume)
    f_merged.CopyInformation(f)
    sitk.WriteImage(f_merged, os.path.join(save_to, fn))
