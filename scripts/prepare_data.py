import numpy as np
import SimpleITK as sitk
import os
import zarr

data_dir = "/export/projects/Candela/datasets/lits_challenge/all/"
save_path = "/export/projects/Candela/datasets/by_project/lits/data.zarr"
zgroup = zarr.open_group(store=save_path, mode='w', path="/")
zarr_kwargs = {'chunks': (1, 512, 512),
               'compressor': zarr.Blosc(cname='lz4', clevel=9, shuffle=1)}

for i in range(130):
    print("Processing volume {}".format(i))
    volume = sitk.ReadImage(os.path.join(data_dir, "volume-"+str(i)+".nii"))
    volume_np = sitk.GetArrayFromImage(volume)
    seg = sitk.ReadImage(os.path.join(data_dir, "segmentation-"+str(i)+".nii"))
    seg_np = sitk.GetArrayFromImage(seg)
    subgroup = zgroup.create_group(str(i))
    print(volume_np.shape)
    subgroup.create_dataset("volume", shape=volume_np.shape, data=volume_np,
                            dtype=np.float32, **zarr_kwargs)
    subgroup.create_dataset("segmentation", shape=seg_np.shape, data=seg_np,
                            dtype=np.int16, **zarr_kwargs)
