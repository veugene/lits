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
    return parser.parse_args()
    
    
def preprocess(volume):
    out = resize_stack(volume.astype(np.float32),
                       size=(256, 256), interp='bilinear')
    out = np.expand_dims(out, 1)
    out /= 255.
    out = np.clip(out, -2., 2.)
    return out


def postprocess(volume):
    print("values ... ", volume.min(), volume.max(), volume.mean())
    out = (volume >= 0.5).astype(np.int16)
    out = np.squeeze(out, axis=1)
    out = resize_stack(out, size=(512, 512), interp='nearest')
    print("prediction: ", volume.shape)
    print("postprocess: ", out.shape)
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
    mask = largest_connected_component(liver_volume>0)
    indices = np.unique(np.where(liver_volume)[0])
    if liver_extent:
        indices = [i for i in range(indices[0]-liver_extent,
                                    indices[-1]+liver_extent)]
    return indices


def load_model(model_path):
    input_varieties = [(None, 'output_1'),
                       ('output_0', 'output_1')]
    
    def get_custom_objects(output_0_name, output_1_name):
        custom_object_list = []
        custom_object_list.append(Dice(2, output_name=output_0_name))
        custom_object_list.append(custom_object_list[-1].get_metrics())
        custom_object_list.append(Dice(2, mask_class=0,
                                        output_name=output_0_name))
        custom_object_list.append(custom_object_list[-1].get_metrics())
        custom_object_list.append(Dice([1, 2], output_name=output_1_name))
        custom_object_list.append(custom_object_list[-1].get_metrics())
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
            try_next_input(iter_inputs)
        except StopIteration:
            print("Failed to instantiate the correct set of custom objects "
                  "to load the model.")
            raise
        return model
        
    iter_inputs = iter(input_varieties)
    model = try_next_input(iter_inputs)
    return model


def predict(volume, model, many_orientations):
    #ops = [(lambda x:x,), (np.fliplr,), (np.flipud,), (np.fliplr, np.flipud),
           #(np.rot90,), (np.rot90, np.fliplr), (np.rot90, np.flipud),
           #(np.rot90, np.fliplr, np.flipud)]
    ops = [(lambda x:x,), (np.fliplr,), (np.flipud,), (np.fliplr, np.flipud)]
    num_ops = 4 if many_orientations else 1
    predictions = []
    for i in range(num_ops):
        op_set = ops[i]
        v = volume.T
        for op in op_set:
            v = op(v)
        pred = model.predict(v.T)
        p = pred.T
        for op in op_set[::-1]:
            p = op(p)
        predictions.append(p.T)
    prediction = np.mean(predictions, axis=0)
    return prediction

    
if __name__=='__main__':
    args = parse()

    if not os.path.exists(args.model_path):
        raise ValueError("model_path doesn't exist: {}"
                         "".format(args.model_path))
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    model = load_model(args.model_path)
    
    ldata_iter = None
    if len(args.liver_data_files):
        ldata_iter = iter(args.liver_data_files)
    for path in args.data_files:
        fn = path.split('/')[-1]
        print("Processing {}".format(path))
        im_f = sitk.ReadImage(path)
        im_np = sitk.GetArrayFromImage(im_f)
        print("im_np: ", im_np.shape)
        input_volume = preprocess(im_np)
        print("input_volume: ", input_volume.shape)
        output_volume = postprocess(predict(input_volume,
                                            model,
                                            args.many_orientations))
        if ldata_iter:
            liver_path = next(ldata_iter)
            liver_f = sitk.ReadImage(liver_path)
            liver_np = sitk.GetArrayFromImage(liver_f)
            indices = select_slices(liver_np, args.liver_extent)
            output_volume[:indices[0]] = 0
            output_volume[indices[-1]:] = 0
        print("output_volume: ", output_volume.shape)
        out_f = sitk.GetImageFromArray(output_volume)
        #out_f.CopyInformation(im_f)
        out_f.SetSpacing(im_f.GetSpacing())
        sitk.WriteImage(out_f, os.path.join(args.output_dir,
                                            fn[:-4]+'.nii.gz'))
            
