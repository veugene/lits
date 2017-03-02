from keras import backend as K
from theano import tensor as T
import numpy as np


def dice_loss(target_class=1, masked_class=None):
    '''
    Dice loss.
    
    Expects integer class labeling in y_true.
    Expects outputs in range [0, 1] in y_pred.
    
    Computes the soft dice loss considering all classes in target_class as one
    aggregate target class and ignoring all elements with ground truth classes
    in masked_class.
    
    target_class : integer or list
    masked_class : integer or list
    '''
    if not hasattr(target_class, '__len__'):
        target_class = [target_class]
    if masked_class is not None and not hasattr(masked_class, '__len__'):
        masked_class = [masked_class]
    
    # Define the keras expression.
    def dice(y_true, y_pred):
        smooth = 1
        y_true_f = K.flatten(y_true)
        y_true_f = K.cast(y_true_f, 'int32')
        y_pred_f = K.flatten(y_pred)
        y_target = K.sum([K.equal(y_true_f, t) for t in target_class], axis=0)
        if masked_class is not None:
            mask_out = K.sum([K.equal(y_true_f, t) for t in masked_class], 
                             axis=0)
            idxs = K.not_equal(mask_out, 1).nonzero()
            y_target = y_target[idxs]
            y_pred_f = y_pred_f[idxs]
        intersection = K.sum(y_target * y_pred_f)
        return -(2.*intersection+smooth) / \
                (K.sum(y_target)+K.sum(y_pred_f)+smooth)
    
    # Set a custom function name
    tag = "_"+"_".join(str(i) for i in target_class)
    if masked_class is not None:
        tag += "_"+"_".join("m"+str(i) for i in masked_class)
    dice.__name__ = "dice_loss"+tag
    
    return dice
