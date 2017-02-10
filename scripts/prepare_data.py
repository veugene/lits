import numpy as np
import SimpleITK as sitk
import os
import zarr

data_dir = "/export/projects/Candela/datasets/lits_challenge/all/"
save_dir = "/export/projects/Candela/datasets/by_project/lits/"
save_name = "data"
include_classes = [2]
if len(include_classes):
    save_name += "_"+"_".join(str(i) for i in include_classes)
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
    
    if len(include_classes):
        for c in include_classes:
            indices = np.where(seg_np==c)
            volume_np = volume_np[np.unique(indices[0])]
            seg_np = seg_np[np.unique(indices[0])]
    if len(volume_np)==0:
        print("WARNING! Skipping empty volume #{}".format(i))
        continue
    
    print("Saving {} slices".format(volume_np.shape[0]))
    subgroup = zgroup.create_group(str(i))
    subgroup.create_dataset("volume", shape=volume_np.shape, data=volume_np,
                            dtype=np.float32, **zarr_kwargs)
    subgroup.create_dataset("segmentation", shape=seg_np.shape, data=seg_np,
                            dtype=np.int16, **zarr_kwargs)
