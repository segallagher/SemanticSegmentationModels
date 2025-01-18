from keras import losses, optimizers, metrics, models, callbacks, src
import tensorflow as tf
from pathlib import Path
from utils import load_data, create_autoencoder, DiceCoefficient, LogBestEpoch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

curr_dir = Path(__file__).resolve().parent

# Get Hyperparameters
hyperparamPath = curr_dir / "hyperparameters.json"
hyperparam = None

try:
    with open(hyperparamPath, 'r') as file:
        # Load Hyperparameters
        hyperparam = json.load(file)
        # Convert json encoded map to python
        hyperparam["color_mapping"] = converted_mapping = {tuple(map(int, key.split(','))): value for key, value in hyperparam["color_mapping"].items()}
        # Assign num_classes based off color_mapping
        hyperparam["num_classes"] = len(hyperparam["color_mapping"])
except FileNotFoundError:
    print("Hyperparameters file not found")
except json.JSONDecodeError:
    print("Error decoding hyperparameters from json")
except Exception as e:
    print(f"Error when reading hyperparameters: {e}")

# Load Data
data_dir = Path(hyperparam["data_path"]).resolve()
train_images, train_labels, val_images, val_labels, _ = load_data(data_dir, hyperparam["num_classes"], hyperparam["color_mapping"])


# Allow GPU memory growth to avoid running out of GPU memory
# code from https://www.tensorflow.org/guide/gpu
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# Create model metrics
meaniou_metric = metrics.OneHotMeanIoU(num_classes=hyperparam["num_classes"])
dice_coef = DiceCoefficient()
optimizer = optimizers.Adam(learning_rate=hyperparam["learning_rate"])

# Load model if model_path is provided, useful for continuing training
# Otherwise create a new model
model = None
if hyperparam.get("model_path"):
    model: src.Functional = models.load_model(hyperparam["model_path"], custom_objects={"DiceCoefficient": dice_coef})
else:
    model = create_autoencoder(hyperparam["input_shape"], hyperparam["num_classes"],
                   hyperparam["encoder_layer_sizes"], hyperparam["bottleneck_size"], hyperparam["decoder_layer_sizes"],
                   hyperparam["conv_per_block"], hyperparam["kernel_size"], hyperparam["activation"], hyperparam["padding"])
model.compile(optimizer=optimizer, loss=losses.categorical_crossentropy, metrics=['accuracy', meaniou_metric, dice_coef])
model.summary()

# Callbacks
earlystopping = callbacks.EarlyStopping(monitor="val_one_hot_mean_io_u", patience=10, mode="max")
checkpoint = callbacks.ModelCheckpoint(filepath=hyperparam["output_name"], monitor="val_one_hot_mean_io_u", save_best_only=True, mode="max", verbose=1)
lr_scheduler = callbacks.ReduceLROnPlateau(monitor="val_one_hot_mean_io_u", factor=0.9, patience=6)
log_best_epoch = LogBestEpoch('val_one_hot_mean_io_u', ['accuracy', "dice_coefficient"], output_name="autoencoder_metrics.json")
tensorboard = callbacks.TensorBoard(log_dir="logs/autoencoder", histogram_freq=1)


# Get training parameters from hyperparameters
initial_epoch = 0
if hyperparam.get("initial_epoch"):
    initial_epoch = hyperparam["initial_epoch"]

batch_size = 2
if hyperparam.get("batch_size"):
    initial_epoch = hyperparam["batch_size"]

max_epochs = 5000
if hyperparam.get("max_epochs"):
    initial_epoch = hyperparam["max_epochs"]

# Train Model
model.fit(x=train_images, y=train_labels, 
            epochs=max_epochs, batch_size=batch_size, callbacks=[earlystopping, checkpoint, lr_scheduler, log_best_epoch, tensorboard],
            shuffle=True, validation_data=(val_images, val_labels), initial_epoch=initial_epoch)

# Get the model's predictions
batch_size = 1
val_pred = []
val_labels_indicies = []

for i in range(0, len(val_images), batch_size):
    batch_images = val_images[i:i+batch_size]
    batch_labels = val_labels[i:i+batch_size]
    
    # Predict and get argmax for the current batch
    batch_pred = np.argmax(model.predict(batch_images, verbose=0), axis=-1)
    batch_labels_indicies = np.argmax(batch_labels, axis=-1)
    
    # Append the results
    val_pred.append(batch_pred)
    val_labels_indicies.append(batch_labels_indicies)
val_pred_flat = np.concatenate(val_pred).flatten()
val_labels_indicies_flat = np.concatenate(val_labels_indicies).flatten()

# Make confusion matrix and save as png
cm = tf.math.confusion_matrix(val_labels_indicies_flat, val_pred_flat)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=np.arange(hyperparam["num_classes"]), yticklabels=np.arange(hyperparam["num_classes"]))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Autoencoder Confusion Matrix')
plt.tight_layout()

plt.savefig('autoencoder_confusion_matrix.png')