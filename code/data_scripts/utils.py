
import os
import numpy as np
from PIL import Image
import glob


# Load preprocessed data
def load_dir(directory:str, num_classes:int, color_to_class_map:dict) -> tuple[np.ndarray, np.ndarray]:
    img_dir = os.path.join(directory, "images")
    img_list = []
    for image in os.listdir(img_dir):
        # Path to image
        img_path = os.path.join(img_dir, image)
        with Image.open(img_path) as img:
            # Convert image to np array
            img_array = np.array(img)
            img_list.append(img_array)

    lab_dir = os.path.join(directory, "labels")
    lab_list = []
    if os.path.exists(lab_dir):
        for image in os.listdir(lab_dir):
            # Path to image
            img_path = os.path.join(lab_dir, image)
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


def open_and_resize_images(directory:str, size:tuple=(256,256), interpolation:int=Image.NEAREST) -> np.ndarray:
    image_paths = glob.glob(os.path.join(directory, "*.png"))
    image_list = []
    for index in range(len(image_paths)):
        image = Image.open(image_paths[index]).convert('RGB')
        resized_image = image.resize(size, interpolation)
        image_list.append(resized_image)
    return np.array(image_list).astype(np.uint8)


def process_directory(input_directory:str, output_directory:str, size:tuple=(256,256), color_channels:int=3) -> None:
    resized_images = np.zeros((0, *size, color_channels), dtype=np.uint8)
    resized_labels = np.zeros((0, *size, color_channels), dtype=np.uint8)

    for sequence in os.listdir(input_directory):
        sequence_dir = os.path.join(input_directory, sequence)
        img_path = os.path.join(sequence_dir, "images")
        label_path = os.path.join(sequence_dir, "labels")

        # Process images
        resized_images = np.append(resized_images, open_and_resize_images(img_path, tuple(size), Image.BICUBIC), axis=0).astype(np.uint8)
        
        # Check if label directory exists before processing labels
        if os.path.exists(label_path):
            resized_labels = np.append(resized_labels, open_and_resize_images(label_path, tuple(size), Image.NEAREST), axis=0).astype(np.uint8)
    
    
    # Export images 
    os.makedirs(os.path.join(output_directory, "images"), exist_ok=True)
    for i, image in enumerate(resized_images):
        Image.fromarray(image).save(os.path.join(output_directory, "images", f"{i}.png"))
    
    if resized_labels.shape[0] > 0:
        os.makedirs(os.path.join(output_directory, "labels"), exist_ok=True)
        for i, image in enumerate(resized_labels):
            Image.fromarray(image).save(os.path.join(output_directory, "labels", f"{i}.png"))
    print(f"Processed {output_directory}")

def segmap_to_image(segmaps:np.ndarray, class_to_color_map:dict, output_dir:str=os.getcwd(), color_channels:int=3, filename:str=None):
    for i, segmap in enumerate(segmaps):

        # Get the class with the highest probability
        argmax_labels = np.argmax(segmap, axis=-1)
        
        # convert labels to colors
        image_arr = np.zeros((segmap.shape[0],segmap.shape[1], color_channels), dtype=np.uint8)
        for label, color in class_to_color_map.items():
            image_arr[argmax_labels == label] = color

        # turn array into image
        image = Image.fromarray(image_arr)

        # Save image
        if filename:
            image.save(os.path.join(output_dir, f"{filename}"))
        else:
            image.save(os.path.join(output_dir, f"{i}.png"))
        

