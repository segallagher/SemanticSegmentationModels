from tensorflow.keras import callbacks, backend as K, metrics
from keras import layers, models
from pathlib import Path
from PIL import Image
import numpy as np
import json
import numpy as np

# Data loader
# Loads data from a file structure like the one found on UAVid dataset
def load_data(directory:Path, num_classes:int, color_to_class_map:dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_dir = directory / "train"
    val_dir = directory / "val"
    test_dir = directory / "test"

    train_images, train_labels = load_dir(train_dir, num_classes, color_to_class_map)
    val_images, val_labels = load_dir(val_dir, num_classes, color_to_class_map)
    test_images, _ = load_dir(test_dir, num_classes, color_to_class_map)
    return train_images, train_labels, val_images, val_labels, test_images

# loads a directory from the UAVid dataset
def load_dir(directory:Path, num_classes:int, color_to_class_map:dict) -> tuple[np.ndarray, np.ndarray]:
    img_dir = directory / "images"
    img_list = []
    for image in img_dir.iterdir():
        # Path to image
        img_path =img_dir / image
        with Image.open(img_path) as img:
            # Convert image to np array
            img_array = np.array(img)
            img_list.append(img_array)

    lab_dir = directory / "labels"
    lab_list = []
    if lab_dir.exists():
        for image in lab_dir.iterdir():
            # Path to image
            img_path = lab_dir / image
            with Image.open(img_path) as img:
                # Convert image to array
                img_array = np.array(img)

                # Make emtpy labeled image
                labeled_img = np.zeros(shape=(img_array.shape[0], img_array.shape[1], num_classes), dtype=np.uint8)

                # Start encoding masks of images
                for color, label in color_to_class_map.items():
                    # Generate mask of the current label
                    mask = np.all(img_array == np.array(color), axis=-1)

                    # One-hot encode
                    labeled_img[mask, label] = 1
                lab_list.append(labeled_img)
    
    img_arr = None
    if len(img_list) > 0:
        img_arr = np.stack(img_list)
    lab_arr = None
    if len(lab_list) > 0:
        lab_arr = np.stack(lab_list)
    return img_arr, lab_arr


# Create model
def fcn_encoder_block(kernel_size:tuple, activation:str, layer_size:int, append_layer, num_conv:int=1, padding:str='same'):
    x = layers.Conv2D(layer_size, kernel_size=kernel_size, padding=padding, activation=None, kernel_initializer='he_normal')(append_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    for _ in range(num_conv-1):
        x = layers.Conv2D(layer_size, kernel_size=kernel_size, padding=padding, activation=None, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)
    return x

def fcn_bottleneck(kernel_size:tuple, activation:str, layer_size:int, append_layer, num_conv:int=1, padding:str='same'):
    x = layers.Conv2D(layer_size, kernel_size=kernel_size, padding=padding, activation=None, kernel_initializer='he_normal')(append_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    for _ in range(num_conv - 1):
        x = layers.Conv2D(layer_size, kernel_size=kernel_size, padding=padding, activation=None, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
    return x

def fcn_decoder_block(kernel_size:tuple, activation:str, layer_size:int, append_layer, num_conv:int=1, padding:str='same'):
    x = layers.UpSampling2D(size=(2,2))(append_layer)
    x = layers.Conv2D(layer_size, kernel_size=kernel_size, padding=padding, activation=None, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    for _ in range(num_conv - 1):
        x = layers.Conv2D(layer_size, kernel_size=kernel_size, padding=padding, activation=None, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
    return x

def create_fcn(input_shape, num_classes, 
               encoder_layer_sizes:list, bottleneck_size:int, decoder_layer_sizes:list,
               conv_per_block:int=1, kernel_size:tuple=(3,3), activation:str='relu', padding:str='same'):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    for size in encoder_layer_sizes:
        x = fcn_encoder_block(kernel_size=kernel_size, activation=activation, layer_size=size, append_layer=x, num_conv=conv_per_block, padding=padding)

    x = fcn_bottleneck(kernel_size=kernel_size, activation=activation, layer_size=bottleneck_size, append_layer=x, num_conv=conv_per_block, padding=padding)

    for size in decoder_layer_sizes:
        x = fcn_decoder_block(kernel_size=kernel_size, activation=activation, layer_size=size, append_layer=x, num_conv=conv_per_block, padding=padding)

    decoder_output = layers.Conv2D(num_classes, kernel_size=kernel_size, activation='softmax', padding=padding)(x)

    model = models.Model(inputs=inputs, outputs=decoder_output)
    return model

# Metrics
class DiceCoefficient(metrics.Metric):
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


class LogBestEpoch(callbacks.Callback):
    def __init__(self, monitor:str, additional_metrics:list=[], output_name:str="best_metrics.json"):
        super().__init__()
        # Get metric names
        self.monitor = monitor
        self.additional_metrics = additional_metrics
        # Initialize metric values
        self.best_epoch = None
        self.best_monitor_value = -float('inf')
        self.additional_metric_values = [-float('inf') for _ in additional_metrics]
        # Misc
        self.output_name = output_name

    def on_epoch_end(self, epoch, logs=None):
        # Update metric values if epoch's primary monitor value is higher than previous best_monitor_value
        cur_val = logs.get(self.monitor)
        if cur_val is not None and cur_val > self.best_monitor_value:
            self.best_monitor_value = cur_val
            self.best_epoch = epoch + 1
            # update additional metric values
            self.additional_metric_values = [logs.get(metric) for metric in self.additional_metrics]

    def on_train_end(self, logs = None):
        if self.best_epoch is not None:
            # Create data log
            data = {
                "best-epoch": self.best_epoch,
                self.monitor: self.best_monitor_value,
            }
            for i, metric in enumerate(self.additional_metric_values):
                data[f"{self.additional_metrics[i]}"] = metric
            
            with open(self.output_name, 'w') as f:
                json.dump(data, f, indent=4)
