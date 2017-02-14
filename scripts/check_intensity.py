import os
import csv
import numpy as np
import SimpleITK as sitk

'''
This script computes the intensity statistics of labels 1 and 2 from
the LiTS challenge, and saves it as a .csv file.
'''

data_dir = "/export/projects/Candela/datasets/lits_challenge/all/"
save_dir = "/home/imagia/gabriel.chartrand-home/intensity.csv"

def compute_intensity_stats(volume, mask, label):
    stats = {}    
    stats['average_'+str(label)] = np.mean(volume[mask==label])
    stats['std_'+str(label)] = np.std(volume[mask==label])
    return stats

all_stats = [] 
for i in range(130):
    print("Processing volume {}".format(i))
    volume = sitk.ReadImage(os.path.join(data_dir, "volume-"+str(i)+".nii"))
    volume_np = sitk.GetArrayFromImage(volume)
    seg = sitk.ReadImage(os.path.join(data_dir, "segmentation-"+str(i)+".nii"))
    seg_np = sitk.GetArrayFromImage(seg)

    s1 = compute_intensity_stats(volume_np, seg_np, 1)
    s2 = compute_intensity_stats(volume_np, seg_np, 2)
    stats = {}
    stats.update(s1)
    stats.update(s2)
    stats['volume_id'] = "volume-"+str(i)
    all_stats.append(stats)
    
with open(save_dir,'w') as fid:
    w = csv.DictWriter(fid, stats.keys())
    w.writeheader()
    w.writerows(all_stats)
