import argparse
import os
import keras
import glob
import SimpleITK as sitk
import numpy as np
from scipy import ndimage
import sys
sys.path.append("../")
from lib.utils import (resize_stack,
                       consecutive_slice_view)
from lib.callbacks import Dice
from lib.loss import dice_loss

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
    parser.add_argument('--liver_extent',
                        help="how many slices to include adjacent to the "
                             "liver (used with --use_predicted_liver)",
                        required=False, type=int, default=0)
    parser.add_argument('--many_orientations',
                        help="average predictions over all 4 possible flips.",
                        required=False, action='store_true')
    parser.add_argument('--downscale',
                        help="specify whether to downscale inputs to half "
                             "resolution (256, 256)",
                        required=False, action='store_true')
    parser.add_argument('--use_predicted_liver',
                        help="whether to crop to predicted liver volumes ",
                        required=False, action='store_true')
    parser.add_argument('--raw_outputs',
                        help="whether to output raw probability values rather "
                             "than classes",
                        required=False, action='store_true')
    parser.add_argument('--batch_size',
                        help="batch size for prediction",
                        required=False, type=int, default=32)
    return parser.parse_args()
    
    
def preprocess(volume, input_shape, downscale=False):
    if input_shape[-3]>1:
        # 3D or thick input
        volume = consecutive_slice_view(volume, num_consecutive=1)[...]
    out = volume.astype(np.float32)
    if downscale:
        out = resize_stack(out,size=(256, 256), interp='bilinear')
    if len(input_shape)==5 or input_shape[-3]==1:
        # 3D or normal
        out = np.expand_dims(out, 1)
    out /= 255.
    out = np.clip(out, -2., 2.)
    return out


def postprocess(volume, raw=False, downscale=False):
    if raw:
        out = volume.astype(np.float32)
    else:
        out = (volume >= 0.5).astype(np.int16)
    out = np.squeeze(out, axis=1)
    if downscale:
        out = resize_stack(out, size=(512, 512), interp='nearest')
    return out


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
    mask = largest_connected_component(liver_volume>0.5)
    indices = np.unique(np.where(mask)[0])
    if liver_extent:
        min_idx = max(0, indices[0]-liver_extent)
        max_idx = min(len(mask), indices[-1]+liver_extent+1)
        indices = [i for i in range(min_idx, max_idx)]
    return indices, mask


def load_model(model_path):
    sys.setrecursionlimit(99999)
    
    input_varieties = [(None, 'output_1'),
                       ('output_0', 'output_1')]
    
    def get_custom_objects(output_0_name, output_1_name):
        custom_object_list = []
        custom_object_list.append(Dice(2, output_name=output_0_name))
        custom_object_list.extend(custom_object_list[-1].get_metrics())
        custom_object_list.append(Dice(2, mask_class=0,
                                        output_name=output_0_name))
        custom_object_list.extend(custom_object_list[-1].get_metrics())
        custom_object_list.append(Dice([1, 2], output_name=output_1_name))
        custom_object_list.extend(custom_object_list[-1].get_metrics())
        custom_object_list.append(dice_loss(2))
        custom_object_list.append(dice_loss(2, masked_class=0))
        custom_object_list.append(dice_loss([1, 2]))
        custom_objects = dict((f.__name__, f) for f in custom_object_list)
        return custom_objects
    
    def try_next_input(iter_inputs):
        try:
            inputs = next(iter_inputs)
            custom_objects = get_custom_objects(*inputs)
            model = keras.models.load_model(model_path,
                                            custom_objects=custom_objects)
        except ValueError:
            model = try_next_input(iter_inputs)
        except StopIteration:
            print("Failed to instantiate the correct set of custom objects "
                  "to load the model.")
            raise
        except:
            raise
        return model
        
    iter_inputs = iter(input_varieties)
    model = try_next_input(iter_inputs)
    return model


def predict(volume, model, batch_size, many_orientations):
    #ops = [(lambda x:x,), (np.fliplr,), (np.flipud,), (np.fliplr, np.flipud),
           #(np.rot90,), (np.rot90, np.fliplr), (np.rot90, np.flipud),
           #(np.rot90, np.fliplr, np.flipud)]
    ops = [(lambda x:x,), (np.fliplr,), (np.flipud,), (np.fliplr, np.flipud)]
    num_ops = 4 if many_orientations else 1
    num_outputs = len(model.outputs)
    predictions = [None for i in range(num_outputs)]
    for i in range(num_ops):
        print("i: ", i)
        op_set = ops[i]
        v = volume.T
        for op in op_set:
            v = op(v)
        pred = model.predict(v.T, batch_size=batch_size)
        if num_outputs==1:
            pred = [pred]
        for j, p in enumerate(pred):
            p = p.T
            for op in op_set[::-1]:
                p = op(p)
            p = p.T
            if p.ndim==5:
                # consecutive slices; keep middle slice
                p = p[:,:,p.shape[2]//2,:,:]
            if predictions[j] is None:
                predictions[j] = p
            else:
                predictions[j] += p
    for j in range(len(predictions)):
        predictions[j] /= num_ops
    return predictions


if __name__=='__main__':
    args = parse()

    if not os.path.exists(args.model_path):
        raise ValueError("model_path doesn't exist: {}"
                         "".format(args.model_path))
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    model = load_model(args.model_path)
    
    for path in sorted(args.data_files):
        fn = path.split('/')[-1]
        if fn.endswith(".nii"):
            fn = fn[:-4]
        elif fn.endswith(".nii.gz"):
            fn = fn[:-7]
        else:
            raise ValueError("Expecting data files to have .nii or .nii.gz "
                             "extensions.")
        if os.path.exists(os.path.join(args.output_dir,
                                       fn+"_output_0.nii.gz")):
            continue
        
        print("Processing {}".format(path))

        im_f = sitk.ReadImage(path)
        im_np = sitk.GetArrayFromImage(im_f)
        input_volume = preprocess(im_np, 
                                  model.input_shape,
                                  downscale=args.downscale)
        outputs = predict(input_volume, model, args.batch_size,
                          args.many_orientations)
        outputs = [postprocess(out,
                               raw=args.raw_outputs,
                               downscale=args.downscale) for out in outputs]
        if args.use_predicted_liver or not args.raw_outputs:
            indices, liver_mask = select_slices(outputs[1], args.liver_extent)
        if args.use_predicted_liver:
            outputs[0][:indices[0]] = 0
            outputs[0][indices[-1]:] = 0
        if not args.raw_outputs:
            outputs[1][...] = liver_mask
        elif args.use_predicted_liver:
            outputs[1][:indices[0]] = 0
            outputs[1][indices[-1]:] = 0
        for i, output_volume in enumerate(outputs):
            out_f = sitk.GetImageFromArray(output_volume)
            out_f.SetSpacing(im_f.GetSpacing())
            out_f.SetOrigin(im_f.GetOrigin())
            out_f.SetDirection(im_f.GetDirection())
            filename = fn+"_output_{}.nii.gz".format(i)
            sitk.WriteImage(out_f, os.path.join(args.output_dir, filename))
            
