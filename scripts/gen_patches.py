import numpy as np
import SimpleITK as sitk
import os
import zarr
import random
import scipy.misc
from data_tools.patches import create_dataset
from data_tools.binary_morphology import binary_operation
from scipy import ndimage


sample_mask_kwargs = { \
    'lesion_dilation': 2.,   # dilate lesion by this much to get periphery
    'liver_edge_boundary_thickness': 15,
    'fraction_lesion_periphery': 0.6,
    'fraction_liver_edge': 0.2,
    'fraction_random': 0.2,
    'max_lesion_radius': None, # mm
    'nb_workers': 6}
patch_size = 24

# Note: patches are only extracted from liver slices,
#       some of which may not contain lesion.


def get_liver_edge_mask(liver_mask, voxel_spacing, boundary_thickness,
                        nb_workers=None):
    mask_large = binary_operation(input_image=liver_mask,
                                  spacing=voxel_spacing,
                                  radius=boundary_thickness/2.,
                                  operation='dilation',
                                  nb_workers=nb_workers)
    mask_small = binary_operation(input_image=liver_mask,
                                  spacing=voxel_spacing,
                                  radius=boundary_thickness/2.,
                                  operation='erosion',
                                  nb_workers=nb_workers)
    edge_mask = mask_large
    edge_mask[mask_small.astype(np.bool)] = 0
    return edge_mask


def get_lesions_and_periphery(lesion_mask, voxel_spacing, lesion_dilation,
                              max_radius, nb_workers=None):
    labeled_mask, num_lesions = ndimage.measurements.label(lesion_mask)
    small_lesions = np.zeros(lesion_mask.shape, dtype=np.bool)
    large_lesions = np.zeros(lesion_mask.shape, dtype=np.bool)
    small_peri = np.zeros(lesion_mask.shape, dtype=np.bool)
    num_kept = 0
    for i in range(1, num_lesions+1):
        lesion_mask_i = labeled_mask==i
        
        # The effective radius is the mean of the half-width of the bounding
        # box across the three axes.
        slice_x, slice_y, slice_z = ndimage.find_objects(lesion_mask_i)[0]
        bb_dims = np.abs([slice_x.stop-slice_x.start,
                          slice_y.stop-slice_x.start,
                          slice_z.stop-slice_z.start])
        effective_radius = np.mean(bb_dims*np.array(voxel_spacing))/2.
        
        # Find lesion periphery.
        if max_radius is None or effective_radius <= max_radius:
            dilation_radius = effective_radius*(lesion_dilation-1)
            lesion_mask_i_dilated = binary_operation(input_image=lesion_mask_i,
                                                     spacing=voxel_spacing,
                                                     radius=dilation_radius,
                                                     operation='dilation',
                                                     nb_workers=nb_workers)
            lesion_mask_i_periphery = lesion_mask_i_dilated - lesion_mask_i        
            small_lesions[lesion_mask_i] = True
            small_peri[lesion_mask_i_periphery] = True
            num_kept += 1
        else:
            large_lesions[lesion_mask_i] = True
            
    print("{} of {} lesions met {} mm radius criterion"
          "".format(num_kept, num_lesions, max_radius))
            
    # Exclude lesions from lesion periphery masks.
    small_peri[small_lesions] = False
    small_peri[large_lesions] = False
    
    return small_lesions, small_peri


def get_sampling_mask(scan, labels, voxel_spacing,
                      lesion_dilation, liver_edge_boundary_thickness,
                      fraction_lesion_periphery, fraction_liver_edge,
                      fraction_random, max_lesion_radius, nb_workers=None):
    
    liver_edge_mask = get_liver_edge_mask( \
                              liver_mask=(labels==1),
                              voxel_spacing=voxel_spacing,
                              boundary_thickness=liver_edge_boundary_thickness,
                              nb_workers=nb_workers)
    lesion_mask, lesion_periphery_mask = get_lesions_and_periphery( \
                              lesion_mask=(labels==2),
                              voxel_spacing=voxel_spacing,
                              lesion_dilation=lesion_dilation,
                              max_radius=max_lesion_radius,
                              nb_workers=nb_workers)
    
    # Started building combined mask for nonlesion points.
    nonlesion_mask = np.zeros(scan.shape, dtype=np.bool)
    num_lesion_points = np.count_nonzero(lesion_mask)
    def sample_points(mask, fraction):
        indices = np.where(mask)
        num_samples = int(np.ceil(num_lesion_points*fraction))
        R = np.random.permutation(len(indices[0]))[:num_samples]
        indices = tuple([idx_list[R] for idx_list in indices])
        return indices
    
    # Include lesion periphery points.
    indices = sample_points(lesion_periphery_mask,
                            fraction_lesion_periphery)
    nonlesion_mask[indices] = True
    
    # Include points near the liver edges.
    indices = sample_points(liver_edge_mask, fraction_liver_edge)
    nonlesion_mask[indices] = True
    
    # Include other randomly sampled points from the volume.
    # Limit sampling to only the axial slices with liver and regions not found
    # in lesion_periphery_mask and liver_edge_mask.
    liver_slice_indices = np.where(labels==1)[0]
    other_mask = np.zeros(scan.shape, dtype=np.bool)
    other_mask[liver_slice_indices,:,:] = True
    other_mask[lesion_mask] = False
    other_mask[lesion_periphery_mask] = False
    indices = sample_points(other_mask, fraction_random)
    nonlesion_mask[indices] = True
    
    # Generate integer mask
    output_mask = np.zeros(scan.shape, dtype=np.int16)
    output_mask[nonlesion_mask] = 1
    output_mask[lesion_mask] = 2
    
    return output_mask


if __name__=='__main__':
    # Define load and save directories
    data_dir = "/export/projects/Candela/datasets/lits_challenge/all/"
    save_dir = "/data/TransientData/Candela/lits_challenge/patches_24_all/"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i in range(87,130):
        print("Processing volume {}".format(i))
        volume = sitk.ReadImage(os.path.join(data_dir,
                                             "volume-"+str(i)+".nii.gz"))
        volume_np = sitk.GetArrayFromImage(volume)
        seg = sitk.ReadImage(os.path.join(data_dir,
                                          "segmentation-"+str(i)+".nii.gz"))
        seg_np = sitk.GetArrayFromImage(seg)
        spacing = volume.GetSpacing()[::-1]
        assert(volume_np.shape[1:]==(512,512))
        
        mask = get_sampling_mask(scan=volume_np,
                                 labels=seg_np,
                                 voxel_spacing=spacing,
                                 **sample_mask_kwargs)
        
        if np.count_nonzero(mask):
            print("saving patches to file")
            save_path = os.path.join(save_dir, "patch_set_{}.hdf5".format(i))
            create_dataset(save_path=save_path,
                           patchsize=patch_size,
                           volume=volume_np,
                           mask=mask,
                           class_list=[1, 2],
                           random_order=True,
                           batchsize=32,
                           file_format='hdf5',
                           show_progress=True)
            
            mask_sitk = sitk.GetImageFromArray(mask)
            mask_sitk.CopyInformation(seg)
            sitk.WriteImage(mask_sitk,
                            os.path.join(save_dir, "{}_mask.nii.gz".format(i)))
    
