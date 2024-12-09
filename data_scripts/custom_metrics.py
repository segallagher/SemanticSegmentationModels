
import keras.backend as K
from keras.metrics import Metric

# Metrics
class DiceCoefficient(Metric):
    def __init__(self, name='dice_coefficient', smooth=100, **kwargs):
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.smooth = smooth
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.true_sum = self.add_weight(name='true_sum', initializer='zeros')
        self.pred_sum = self.add_weight(name='pred_sum', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Flatten the tensors to 1D for calculation
        y_true_f = K.cast(K.flatten(y_true), dtype='float32')
        y_pred_f = K.cast(K.flatten(y_pred), dtype='float32')
        
        # Calculate intersection and sums for Dice coefficient
        intersection = K.sum(y_true_f * y_pred_f)
        true_sum = K.sum(y_true_f)
        pred_sum = K.sum(y_pred_f)
        
        # Update the state with current batch values
        self.intersection.assign_add(intersection)
        self.true_sum.assign_add(true_sum)
        self.pred_sum.assign_add(pred_sum)

    def result(self):
        # Compute the Dice coefficient
        dice = (2. * self.intersection + self.smooth) / (self.true_sum + self.pred_sum + self.smooth)
        return dice

    def reset_state(self):
        # Reset the weights at the start of each epoch
        self.intersection.assign(0.)
        self.true_sum.assign(0.)
        self.pred_sum.assign(0.)
