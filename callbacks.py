from keras.callbacks import Callback
from keras import backend as K
import theano
import numpy as np

class FullDice(Callback):
    """
    Compute the dice over the full set of data.
    
    Expects the following metrics: d_I, d_A, d_B
    d_I : number of pixels in the intersection of masks A and B
    d_A : number of pixels in mask A
    d_B : number of pixels in mask B
    
    Also requires a dummy 'fdice' metric to record this callback's output.
    
    The functions for these metrics can be retrieved via FullDice.get_metrics()
    """
    
    @staticmethod
    def get_metrics(target_class):
        '''
        Get keras metrics that compute the values consumed by this callback 
        (d_I, d_A, dB) and a dummy metric that serves only to create an 
        'fdice' keyword entry for storing the computed dice value.
        
        All intermediate metrics are divided by the number of samples to
        counterbalance the way keras collects metrics across batches (by
        averaging across all samples).
        '''
        def full_dice_metrics(y_true, y_pred):
            ''' Temporary metrics used by the FullDice callback. '''
            y_true_f = K.flatten(y_true)
            y_true_f = K.cast(y_true_f, 'int32')
            y_true_f = K.equal(y_true_f, target_class)
            y_pred_f = K.flatten(y_pred)
            n = y_true.shape[0]
            return {'d_I': K.sum(y_true_f * y_pred_f)/n,
                    'd_A': K.sum(y_true_f)/n,
                    'd_B': K.sum(y_pred_f)/n,
                    'fdice': theano.tensor.as_tensor_variable(0)}
        return full_dice_metrics
    
    def on_epoch_begin(self, epoch, logs=None):
        self.totals = {'d_I': 0,
                       'd_A': 0,
                       'd_B': 0}
        
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        
        # Record d_I, d_A, d_B
        for k in self.totals.keys():
            if k not in logs:
                raise ValueError("FullDice callback expects metrics "
                                "d_I, d_A, d_B but {} is missing."
                                "".format(k))
            if k not in self.totals:
                self.totals[k] = 0
            self.totals[k] += logs[k]
            
        # Compute dice
        dice = 1.
        if self.totals['d_A'] or self.totals['d_B']:
            dice = self.compute_dice(I=self.totals['d_I'],
                                     A=self.totals['d_A'],
                                     B=self.totals['d_B'])
        
        # Update the dictionary (write the dice metric and remove used metrics)
        logs['fdice'] = dice
        for k in self.totals.keys():
            logs.pop(k)
        self.dice = {'fdice': dice}
        
        
    def on_epoch_end(self, epoch, logs=None):
        # Update logs; if validation values exist, compute validation dice.
        logs = logs or {}
        val_totals = {}
        logs.update(self.dice)
        for k in self.totals.keys():
            logs.pop(k)
            if 'val_'+k in logs:
                val_totals[k] = logs.pop('val_'+k)
        if len(val_totals):
            val_dice = self.compute_dice(I=val_totals['d_I'],
                                         A=val_totals['d_A'],
                                         B=val_totals['d_B'])
            logs['val_fdice'] = val_dice
            
    def compute_dice(self, I, A, B):
        if not (A or B):
            return 1.
        else:
            return 2*I/float(A+B)
