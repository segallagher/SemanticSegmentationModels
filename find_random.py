from autoencoder.utils import get_hyperparam, load_data
from pathlib import Path
import numpy as np

def get_mean_iou(pred, truth, num_class) -> float:
    # Initialize IoU sum
    iou_sum = 0

    # Iterate through each class
    for class_id in range(num_class):
        # Create boolean masks for the ground truth and prediction
        gt_class = (truth == class_id)
        pred_class = (pred == class_id)
        
        # Compute intersection and union
        intersection = np.sum(gt_class & pred_class)
        union = np.sum(gt_class | pred_class)
        
        # Compute IoU for the current class
        if union != 0:
            iou = intersection / union
        else:
            iou = 0  # Avoid division by zero if there's no union
        
        # Add to IoU sum
        iou_sum += iou
    return float(iou_sum / num_class)


hyperparam = get_hyperparam()
data_dir = Path("data").resolve()
train_images, train_labels, val_images, val_labels, _ = load_data(data_dir, hyperparam["num_classes"], hyperparam["color_mapping"])

mIoUs = []
for label in train_labels:
    rand_pred = np.random.randint(0,7,(256,256))
    ground_truth = np.argmax(label, axis=2)
    mIoU = get_mean_iou(rand_pred, ground_truth, hyperparam["num_classes"])
    mIoUs.append(mIoU)

print(mIoUs)
print("AVG: ", sum(mIoUs) / len(mIoUs))