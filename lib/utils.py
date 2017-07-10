import zarr
import h5py
from skimage import transform
import numpy as np
from data_tools.data_augmentation import random_transform
from data_tools.io import data_flow
from data_tools.wrap import (multi_source_array,
                             delayed_view)
from keras import backend as K


def resize_stack(arr, size, interp='bilinear'):
    """
    Resize each slice (indexed at dimension 0) of a 3D volume or of each 3D
    volume in a stack of volumes.
    """
    out = np.zeros(arr.shape[:-2]+size, dtype=arr.dtype)
    for idx in np.ndindex(arr.shape[:-2]):
        out[idx] = transform.resize(arr[idx],
                                    output_shape=size,
                                    mode='constant',
                                    cval=0,
                                    clip=True,
                                    preserve_range=True)
    return out

class consecutive_slice_view(delayed_view):
    """
    A modified version of delayed_view which returns a block including 
    num_consecutive slices above and num_consecutive slices below any 
    requested slice (indexed along axis 0).
    
    For example, with num_consecutive=1, when indexing into array A with shape
    (100,50,50), A[25] would return a slice of shape (3, 50, 50) and A[5:10]
    would return a block with shape (5, 3, 50, 50).
    """
    def __init__(self, *args, num_consecutive, **kwargs):
        self.num_consecutive = num_consecutive
        super(consecutive_slice_view, self).__init__(*args, **kwargs)
        
    def _get_element(self, int_key, key_remainder=None):
        if not isinstance(int_key, (int, np.integer)):
            raise IndexError("cannot index with {}".format(type(int_key)))
        idx = self.arr_indices[int_key]
        if key_remainder is not None:
            idx = (idx,)+key_remainder
        idx = int(idx)  # Some libraries don't like np.integer
        n = self.num_consecutive
        if idx-n:
            elem = self.arr[idx-n:idx+n+1]
            ret_arr = np.zeros((2*n+1,)+np.shape(elem)[1:], dtype=self.dtype)
            ret_arr[:len(elem)] = elem
        else:
            elem = self.arr[0:idx+n+1]
            ret_arr = np.zeros((2*n+1,)+np.shape(elem)[1:], dtype=self.dtype)
            ret_arr[-len(elem):] = elem
        return ret_arr
    
    def _get_block(self, values, key_remainder=None):
        item_block = None
        for i, v in enumerate(values):
            # Lists in the aggregate key index in tandem;
            # so, index into those lists (the first list is `values`)
            v_key_remainder = key_remainder
            if isinstance(values, tuple) or isinstance(values, list):
                if key_remainder is not None:
                    broadcasted_key_remainder = ()
                    for k in key_remainder:
                        if hasattr(k, '__len__') and len(k)==np.size(k):
                            broadcasted_key_remainder += (k[i],)
                        else:
                            broadcasted_key_remainder += (k,)
                    v_key_remainder = broadcasted_key_remainder
            
            # Make a single read at an integer index of axis 0
            elem = self._get_element(v, v_key_remainder)
            if item_block is None:
                item_block = np.zeros((len(values),2*self.num_consecutive+1)\
                                      +elem.shape, self.dtype)
            item_block[i] = elem
        return item_block

def data_generator(data_path, volume_indices, batch_size,
                   nb_io_workers=1, nb_proc_workers=0,
                   shuffle=False, loop_forever=False, downscale=False,
                   transform_kwargs=None, data_flow_kwargs=None,
                   align_intensity=False, num_consecutive=None, rng=None):
    """
    Open data files, wrap data for access, and set up proprocessing; then,
    initialize the generic data_flow.
    """
    
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
    if num_consecutive is not None:
        msa_vol = consecutive_slice_view(msa_vol,
                                         num_consecutive=num_consecutive)
        msa_seg = consecutive_slice_view(msa_seg,
                                         num_consecutive=num_consecutive)
    
    # Function to rescale the data and do data augmentation, if requested
    def preprocessor(batch):
        b0, b1 = batch
        if align_intensity:
            mean_liver = np.mean(b0[b1==1])
            b0 += 100 - mean_liver
        if downscale:
            b0 = resize_stack(b0, size=(256, 256), interp='bilinear')
            b1 = resize_stack(b1, size=(256, 256), interp='nearest')
        if transform_kwargs is not None:
            for idx in np.ndindex(b0.shape[:-3]):
                x, y = random_transform(b0[idx], b1[idx], **transform_kwargs)
                b0[idx], b1[idx] = x, y
        b0, b1 = np.expand_dims(b0, 1), np.expand_dims(b1, 1)
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
    """
    Return a tuple with the ground truth repeated a custom number of times.
    """
    for batch in flow:
        if num_outputs==1:
            yield batch
        else:
            yield (batch[0], [batch[1] for i in range(num_outputs)])


def load_and_freeze_weights(model, load_path, freeze=True, verbose=False,
                            layers_to_not_freeze=None, freeze_mask=None,
                            load_mask=None, depth_offset=0):
    if load_mask is None:
        load_mask = []
    f = h5py.File(load_path, mode='r')
    if 'layer_names' not in f.attrs and 'model_weights' in f:
        f = f['model_weights']
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
    weights_dict = dict(((w.name, w) for w in model.weights))
    used_names = []
    for name in layer_names:
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        for wname_load in weight_names:
            
            # Skip masked out weights.
            for m in load_mask:
                if m in wname_load:
                    continue
                
            # Rename weights to account for depth offset.
            wname_parts = wname_load.split('_')
            for i, part in enumerate(wname_parts):
                if part.startswith('d') and part[1:].isdigit():
                    wname_parts[i] = 'd'+str(int(part[1:])+depth_offset)
                    break
                if part.startswith('u') and part[1:].isdigit():
                    wname_parts[i] = 'u'+str(int(part[1:])+depth_offset)
                    break
                if wname_load.startswith('long_skip_') and part.isdigit():
                    wname_parts[i] = str(int(part)+depth_offset)
                    break
            wname = '_'.join(wname_parts)
            
            # Set weights
            
            ## TEMP
            #if wname not in weights_dict:
                #wname_parts = wname_load.split('_')
                #new_wname_parts = []
                #for i, part in enumerate(wname_parts):
                    #if not part.isdigit():
                        #new_wname_parts.append(part)
                    #else:
                        #new_wname_parts.append(part)
                        #new_wname_parts.append('1')
                #wname = '_'.join(new_wname_parts)
            ## /TEMP
                
            if wname in weights_dict:
                if wname in used_names:
                    raise ValueError("{} already previously loaded!"
                                     "".format(wname))
                used_names.append(wname)
                if verbose:
                    print("Setting weights {}".format(wname))
                weights_dict[wname].set_value(g[wname_load][...])
            else:
                print("WARNING: {} not found in model (skipped)".format(wname))

    if freeze:
        layers_to_freeze = []
        if layers_to_not_freeze is None:
            layers_to_not_freeze = []
        if freeze_mask is None:
            freeze_mask = []
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
            skip = False
            if lname in layers_to_not_freeze:
                skip = True
            for m in freeze_mask:
                # Skip masked out weights.
                if m in lname:
                    skip = True
            if skip:
                print("(Not freezing layer {})".format(lname))
                continue
            layer = find_layer_in_model(lname, model)
            if layer:
                if verbose:
                    print("Freezing layer {}".format(lname))
                layer.trainable = False
        
