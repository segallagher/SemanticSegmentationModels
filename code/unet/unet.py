from keras import losses, optimizers
from keras.metrics import OneHotMeanIoU
from pathlib import Path
from utils import load_data, create_unet, DiceCoefficient, LogBestEpoch
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import json

curr_dir = Path(__file__).resolve().parent

# Load Hyperparameters
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

# Load data
data_dir = None
if hyperparam.get("data_path"):
    data_dir = Path(curr_dir / hyperparam["data_path"]).resolve()
else:
    data_dir = curr_dir / "data"
train_images, train_labels, val_images, val_labels, test_images = load_data(data_dir, hyperparam["num_classes"], hyperparam["color_mapping"])


model = create_unet(hyperparam["input_shape"], hyperparam["num_classes"],
                   hyperparam["initial_filters"], hyperparam["depth"], hyperparam["conv_per_block"],
                   hyperparam["kernel_size"], hyperparam["activation"], hyperparam["padding"])

meaniou_metric = OneHotMeanIoU(num_classes=hyperparam["num_classes"])
dice_coef = DiceCoefficient()
optimizer = optimizers.Adam(learning_rate=hyperparam["learning_rate"])
model.compile(optimizer=optimizer, loss=losses.categorical_crossentropy, metrics=['accuracy', meaniou_metric, dice_coef])
model.summary()

# Callbacks

earlystopping = EarlyStopping(monitor="one_hot_mean_io_u", patience=3, mode="max")
# checkpoint = ModelCheckpoint(filepath=hyperparam["output_name"], monitor="one_hot_mean_io_u", save_best_only=True, mode="max")
lr_scheduler = ReduceLROnPlateau(monitor="one_hot_mean_io_u", factor=0.5, patience=3)
log_best_epoch = LogBestEpoch('one_hot_mean_io_u', ['accuracy', "dice_coefficient"], output_name="unet_metrics.json")


# Train Model
model.fit(x=train_images, y=train_labels,
            epochs=5000, batch_size=4, callbacks=[earlystopping, log_best_epoch],
            shuffle=True, validation_data=(val_images, val_labels))

# Save Model
print("Training complete, saving model")
model.save(hyperparam["output_name"])