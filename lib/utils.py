import zarr
import h5py
from skimage import transform
import numpy as np
from data_tools.data_augmentation import random_transform
from data_tools.io import data_flow
from data_tools.wrap import multi_source_array
from keras import backend as K


def resize_stack(arr, size, interp='bilinear'):
    out = np.zeros((len(arr),)+tuple(size), arr.dtype)
    for i, arr_slice in enumerate(arr):
        out[i] = transform.resize(arr_slice,
                                  output_shape=(256, 256),
                                  mode='constant',
                                  cval=0,
                                  clip=True,
                                  preserve_range=True)
    return out

def data_generator(data_path, volume_indices, batch_size,
                   nb_io_workers=1, nb_proc_workers=0,
                   shuffle=False, loop_forever=False, downscale=False,
                   transform_kwargs=None, data_flow_kwargs=None,
                   rng=None):
    if rng is None:
        rng = np.random.RandomState()
    
    try:
        zgroup = zarr.open_group(data_path, mode='r')
    except:
        print("Failed to open data: {}".format(data_path))
        raise
    
    # Assemble volumes and corresponding segmentations
    volumes = []
    segmentations = []
    for idx in volume_indices:
        subgroup = zgroup[str(idx)]
        volumes.append(subgroup['volume'])
        segmentations.append(subgroup['segmentation'])
    msa_vol = multi_source_array(source_list=volumes, shuffle=False)
    msa_seg = multi_source_array(source_list=segmentations, shuffle=False)
    
    # Function to rescale the data and do data augmentation, if requested
    def preprocessor(batch):
        b0, b1 = batch
        if downscale:
            b0 = resize_stack(b0, size=(256, 256), interp='bilinear')
            b1 = resize_stack(b1, size=(256, 256), interp='nearest')
        b0, b1 = np.expand_dims(b0, 1), np.expand_dims(b1, 1)
        if transform_kwargs is not None:
            for i in range(len(b0)):
                x, y = random_transform(b0[i], b1[i], **transform_kwargs)
                b0[i], b1[i] = x, y
        # standardize
        b0 /= 255.0
        b0 = np.clip(b0, -2.0, 2.0)
        return (b0, b1)
    
    # Prepare the data iterator
    if data_flow_kwargs is None:
        data_flow_kwargs = {}
    data_gen = data_flow(data=[msa_vol, msa_seg],
                         batch_size=batch_size,
                         nb_io_workers=nb_io_workers,
                         nb_proc_workers=nb_proc_workers,
                         shuffle=shuffle, 
                         loop_forever=loop_forever,
                         preprocessor=preprocessor,
                         rng=rng)
    return data_gen


def repeat_flow(flow, num_outputs):
    for batch in flow:
        if num_outputs==1:
            yield batch
        else:
            yield (batch[0], [batch[1] for i in range(num_outputs)])


def load_and_freeze_weights(model, load_path, freeze=True, verbose=False,
                            layers_to_not_freeze=None):
    f = h5py.File(load_path, mode='r')
    if 'layer_names' not in f.attrs and 'model_weights' in f:
        f = f['model_weights']
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
    weights_dict = dict(((w.name, w) for w in model.weights))
    for name in layer_names:
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        for wname in weight_names:
            if wname in weights_dict:
                if verbose:
                    print("Setting weights {}".format(wname))
                weights_dict[wname].set_value(g[wname][...])
    
    if freeze:
        layers_to_freeze = []
        if layers_to_not_freeze is None:
            layers_to_not_freeze = []
        param_names = ["_W", "_b", "_gamma", "_beta",
                       "_running_mean", "_running_std"]
        for name in layer_names:
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            for wname in weight_names:
                for pname in param_names:
                    if wname.endswith(pname):
                        layers_to_freeze.append(wname[:-len(pname)])
        
        def find_layer_in_model(name, model):
            for layer in model.layers:
                if layer.name==name:
                    return layer
                elif layer.__class__.__name__ == 'Model':
                    layer = find_layer_in_model(name, layer)
                    if layer:
                        return layer
            return None
        
        for lname in sorted(set(layers_to_freeze)):
            if lname in layers_to_not_freeze:
                continue
            layer = find_layer_in_model(lname, model)
            if layer:
                if verbose:
                    print("Freezing layer {}".format(lname))
                layer.trainable = False
        
