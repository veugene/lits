import zarr
import h5py
from skimage import transform
import numpy as np
from data_tools.data_augmentation import random_transform
from data_tools.io import data_flow
from data_tools.wrap import (multi_source_array,
                             delayed_view)
from keras import backend as K
from keras.initializers import he_normal


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
                item_block = np.zeros((len(values),)+elem.shape, self.dtype)
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
        if hasattr(num_consecutive, '__len__'):
            if len(num_consecutive)!=2:
                raise ValueError("num_consecutive must be a length two list "
                                 "or an integer. Recieved {}"
                                 "".format(num_consecutive))
            vol_num_consecutive, seg_num_consecutive = num_consecutive
        else:
            vol_num_consecutive = seg_num_consecutive = num_consecutive
        if vol_num_consecutive is not None:
            msa_vol = consecutive_slice_view(msa_vol,
                                           num_consecutive=vol_num_consecutive)
        if seg_num_consecutive is not None:
            msa_seg = consecutive_slice_view(msa_seg,
                                           num_consecutive=seg_num_consecutive)
    
    # Function to rescale the data and do data augmentation, if requested
    def preprocessor(batch):
        b0, b1 = batch
        if align_intensity:
            mean_liver = np.mean(b0[b1==1])
            b0 += 100 - mean_liver
        if downscale:
            b0 = resize_stack(b0, size=(256, 256), interp='bilinear')
            b1 = resize_stack(b1, size=(256, 256), interp='nearest')
        b0, b1 = np.expand_dims(b0, 1), np.expand_dims(b1, 1)
        if transform_kwargs is not None:
            for idx0, idx1 in zip(np.ndindex(b0.shape[:-3]),
                                  np.ndindex(b1.shape[:-3])):
                x, y = random_transform(b0[idx0], b1[idx1], **transform_kwargs)
                b0[idx0], b1[idx1] = x, y
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


class volume_generator(object):
    def __init__(self, data_path, volume_indices,
                 nb_io_workers=1, nb_proc_workers=0, downscale=False, 
                 return_vol_idx=False, num_consecutive=None):
        self.data_path = data_path
        self.volume_indices = volume_indices
        self.nb_io_workers = nb_io_workers
        self.nb_proc_workers = nb_proc_workers
        self.downscale = downscale
        self.return_vol_idx = return_vol_idx
        self.num_consecutive = num_consecutive
        
        try:
            zgroup = zarr.open_group(data_path, mode='r')
        except:
            print("Failed to open data: {}".format(data_path))
            raise
        
        # Assemble volumes and corresponding segmentations
        self.volumes = []
        self.segmentations = []
        for idx in self.volume_indices:
            subgroup = zgroup[str(idx)]
            self.volumes.append(subgroup['volume'])
            self.segmentations.append(subgroup['segmentation'])
        
        # Length
        self.num_volumes = len(self.volumes)
        assert(len(self.segmentations)==self.num_volumes)
            
    def __len__(self):
        return self.num_volumes
    
    def preprocess(self, batch):
        b0, b1 = batch
        if self.downscale:
            b0 = resize_stack(b0, size=(256, 256), interp='bilinear')
            b1 = resize_stack(b1, size=(256, 256), interp='nearest')
        b0, b1 = np.expand_dims(b0, 1), np.expand_dims(b1, 1)
            
        # standardize
        b0 /= 255.0
        b0 = np.clip(b0, -2.0, 2.0)
        
        return (b0, b1)
    
    def flow(self):
        for i in range(self.num_volumes):
            batch = [self.volumes[i], self.segmentations[i]]
            if self.num_consecutive is not None:
                batch = (consecutive_slice_view(
                               batch[0], num_consecutive=self.num_consecutive),
                         consecutive_slice_view(
                               batch[1], num_consecutive=self.num_consecutive))
            data_gen = data_flow(data=batch,
                                 batch_size=1,
                                 nb_io_workers=self.nb_io_workers,
                                 nb_proc_workers=self.nb_proc_workers,
                                 shuffle=False, 
                                 loop_forever=False,
                                 preprocessor=self.preprocess)
            batch = next(data_gen.flow())
            #batch = self.preprocess(batch)
            if self.return_vol_idx:
                batch = (batch[0], batch[1], self.volume_indices[i])
            yield batch


def repeat_flow(flow, num_outputs, adversarial=False):
    """
    Return a tuple with the ground truth repeated a custom number of times.
    """
    assert(num_outputs>=1)
    for batch in flow:
        assert(len(batch)>=2)
        inputs = [batch[0]]
        outputs = []
        for i in range(num_outputs):
            if adversarial:
                # Discriminator inputs.
                inputs.append(batch[1]==(num_outputs-i))
            # All model outputs.
            outputs.append(batch[1])
        
        # Discriminator outputs.
        if adversarial:
            bs = len(batch[0])
            outputs.extend([np.zeros(bs, dtype=np.int32)] * num_outputs)
            outputs.extend([np.ones(bs, dtype=np.int32)] * num_outputs)
            outputs.extend([np.ones(bs, dtype=np.int32)] * num_outputs)
    
        if len(inputs)==1:
            inputs = inputs[0]
        if len(outputs)==1:
            outputs = outputs[0]
            
        yield (inputs, outputs)+batch[2:]
        

def load_and_freeze_weights(model, load_path, freeze=True, verbose=True,
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
            
            # Rename 'bn' weights to 'norm' weights (to convert old models).
            wname_parts = wname_load.split('_')
            for i, part in enumerate(wname_parts):
                if part.startswith('bn'):
                    wname_parts[i] = 'norm'+part[2:]
            wname = '_'.join(wname_parts)
            
            # Skip weights that are not in the model.
            if not wname in weights_dict:
                print("WARNING: {} not found in model (skipped)".format(wname))
                continue
            
            # Skip masked out weights.
            skip = False
            for m in load_mask:
                if m in wname_load:
                    print("(Not setting weights {})".format(wname_load))
                    skip = True
            if skip:
                continue
            
            # Weight names must be unique!.
            if wname in used_names:
                raise ValueError("{} already previously loaded!".format(wname))
            used_names.append(wname)
            
            # Set weights
            if verbose:
                print("Setting weights {}".format(wname))
            var = g[wname_load][...]
            if weights_dict[wname].ndim!=var.ndim:
                # Load 2D into 3D
                weight_shape = tuple(weights_dict[wname].shape.eval())
                #var_z = np.array(he_normal()(weight_shape).eval(),
                                    #dtype=np.float32)
                var_z = np.zeros(weight_shape, dtype=np.float32)
                var_z[weights_dict[wname].shape[0].eval()//2] = var
                var = var_z 
                #var = np.repeat(np.expand_dims(var, axis=0),
                                #repeats=weights_dict[wname].shape[0].eval(),
                                #axis=0).astype(np.float32)
            weights_dict[wname].set_value(var)

    if freeze:
        layers_to_freeze = []
        if layers_to_not_freeze is None:
            layers_to_not_freeze = []
        if freeze_mask is None:
            freeze_mask = []
        param_names = ["/kernel", "/bias", "/gamma", "/beta",
                       "/moving_mean", "/moving_variance"]
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
        
