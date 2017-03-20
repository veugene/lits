import argparse
import os
import keras
import glob
import SimpleITK as sitk
import numpy as np
from scipy import ndimage
import sys
sys.path.append("../")
from lib.utils import resize_stack
from lib.callbacks import Dice
from lib.loss import dice_loss
from data_tools.patches import patch_generator
import progressbar


def parse():
    parser =argparse.ArgumentParser(description="Generate volumes with "
                                    "predictions using a loaded model.")
    parser.add_argument('--data_files',
                        help="paths to input volumes",
                        required=True, nargs='+', type=str)
    parser.add_argument('--model_path',
                        help="path to the saved keras model",
                        required=True, type=str)
    parser.add_argument('--output_dir',
                        help="directory to write prediction volumes to",
                        required=True, type=str)
    parser.add_argument('--liver_data_files',
                        help="paths to volumes with liver labels (used to "
                             "limit prediction to slices that contain liver); "
                             "files should have the same names as in the " "data_dir",
                        required=False, nargs='+', type=str)
    parser.add_argument('--liver_extent',
                        help="how many slices to include adjacent to the "
                             "liver",
                        required=False, type=int,
                        default=0)
    parser.add_argument('--many_orientations',
                        help="average predictions over all 8 possible flips "
                             "and rotations of each slice",
                        required=False, action='store_true')
    parser.add_argument('--patchsize',
                        help="the edge size of the square 2D patches over "
                             "which to predict",
                        required=True, type=int)
    parser.add_argument('--batchsize',
                        help="the batch size for patch prediction",
                        required=False, type=int, default=100000)
    return parser.parse_args()
    
    
def preprocess(volume):
    out = volume.copy().astype(np.float32)
    out /= 255.
    out = np.clip(out, -2., 2.)
    return out


def postprocess(volume):
    out = (volume >= 0.5).astype(np.int16)
    out = np.squeeze(out, axis=1)
    return out


#def select_slices(liver_volume, liver_extent):
    #"""
    #Find the largest connected component in the liver prediction volume and 
    #determine which axial slices include this component. Return the indices
    #for these slices as well as of `liver_extent` slices adjacent to the liver,
    #above and below.
    
    #All nonzero values in the liver_volume are taken as liver labels.
    #"""
    #mask = largest_connected_component(liver_volume>0.5)
    #indices = np.unique(np.where(liver_volume)[0])
    #if liver_extent:
        #indices = [i for i in range(indices[0]-liver_extent,
                                    #indices[-1]+liver_extent)]
    #return indices, mask


def predict(volume, model, patchsize, batchsize,
            liver_volume, many_orientations):
    #ops = [(lambda x:x,), (np.fliplr,), (np.flipud,), (np.fliplr, np.flipud),
           #(np.rot90,), (np.rot90, np.fliplr), (np.rot90, np.flipud),
           #(np.rot90, np.fliplr, np.flipud)]
    ops = [(lambda x:x,), (np.fliplr,), (np.flipud,), (np.fliplr, np.flipud)]
    num_ops = 4 if many_orientations else 1
    predictions = []
    for i in range(num_ops):
        op_set = ops[i]
        v = volume.T
        if liver_volume is not None:
            l = liver_volume.T
        for op in op_set:
            v = op(v)
            if liver_volume is not None:
                l = op(l)
        transformed_volume = v.T
        if liver_volume is not None:
            transformed_liver = l.T
        else:
            transformed_liver = None
        pred = predict_patchwise(volume=transformed_volume, model=model,
                                 patchsize=patchsize, batchsize=batchsize,
                                 liver_volume=transformed_liver)
        p = pred.T
        for op in op_set[::-1]:
            p = op(p)
        predictions.append(p.T)
    prediction = np.mean(predictions, axis=0)
    return prediction


def predict_patchwise(volume, model, patchsize, batchsize, liver_volume=None):
    
    # Mask what needs to be predicted
    process_mask = liver_volume
    if process_mask is None:
        process_mask = np.ones(volume.shape, dtype=np.bool)
    loc = np.where(process_mask)
    
    # Extract patches, and create classification map
    print(" __ Extracting image patches and building classification map ...")
    piter = patch_generator(patchsize=patchsize,
                            source=volume,
                            binary_mask=process_mask)
    print("    ( %r patches to classify)" % len(piter))
    prediction_volume = np.zeros(volume.shape, dtype=np.float32)
    collected = 0
    batchnum = 0
    patches = np.zeros((batchsize,1,patchsize,patchsize), dtype=np.float32)
    bar = progressbar.ProgressBar(maxval=len(piter)).start()
    
    for i, patch in enumerate(piter):
        patches[collected%batchsize,0,:,:] = patch
        collected += 1

        # Classify
        if (collected%batchsize==0 and collected>0) or collected==len(piter):
            batch_input = patches[:collected-batchnum*batchsize]
            prediction = model.predict_on_batch(batch_input)
            #print("DEBUG: ", np.mean(prediction>0.5), len(piter), collected, collected%batchsize)
            i1 = batchsize*batchnum
            i2 = batchsize*(batchnum+1)
            prediction_volume[loc[0][i1:i2], loc[1][i1:i2], loc[2][i1:i2]] = \
                np.squeeze(prediction.copy())
            batchnum += 1
        bar.update(bar.currval+1)
    bar.finish()
    
    return prediction_volume



if __name__=='__main__':
    args = parse()

    if not os.path.exists(args.model_path):
        raise ValueError("model_path doesn't exist: {}"
                         "".format(args.model_path))
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    model = keras.models.load_model(args.model_path)
    
    ldata_iter = None
    if args.liver_data_files is not None:
        if not hasattr(args.liver_data_files, '__len__'):
            ldata_iter = iter([args.liver_data_files])
        else:
            ldata_iter = iter(args.liver_data_files)
    for path in args.data_files:
        fn = path.split('/')[-1]
        print("Processing {}".format(path))
        im_f = sitk.ReadImage(path)
        im_np = sitk.GetArrayFromImage(im_f)
        input_volume = preprocess(im_np)
        liver_volume = None
        if ldata_iter:
            liver_path = next(ldata_iter)
            liver_f = sitk.ReadImage(liver_path)
            liver_volume = sitk.GetArrayFromImage(liver_f)
        output_volume = predict(volume=input_volume,
                                patchsize=args.patchsize,
                                batchsize=args.batchsize,
                                model=model,
                                many_orientations=args.many_orientations, liver_volume=liver_volume)
        #output_volume = postprocess(output_volume)
        out_f = sitk.GetImageFromArray(output_volume)
        out_f.SetSpacing(im_f.GetSpacing())
        out_f.SetOrigin(im_f.GetOrigin())
        out_f.SetDirection(im_f.GetDirection())
        filename = fn[:-4]+".nii.gz"
        sitk.WriteImage(out_f, os.path.join(args.output_dir, filename))
            
