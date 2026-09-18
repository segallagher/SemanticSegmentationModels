from pathlib import Path
import argparse
import random

from PIL import Image


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


def build_augmented_pair(image, label):
    """
    Apply synchronized augmentations to an image and segmentation label.

    Image:
        RGB image

    Label:
        RGB semantic-segmentation mask
    """

    # # Horizontal flip
    # if random.random() < 1.5:
    #     image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    #     label = label.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    # # Small arbitrary rotation
    # if random.random() < 0.5:
    #     angle = random.uniform(-15, 15)

    #     image = image.rotate(
    #         angle,
    #         resample=Image.Resampling.BILINEAR,
    #         expand=False,
    #         fillcolor=(0, 0, 0),
    #     )

    #     label = label.rotate(
    #         angle,
    #         resample=Image.Resampling.NEAREST,
    #         expand=False,
    #         fillcolor=(0, 0, 0),
    #     )

    # Random crop followed by resize
    if random.random() < 1.5:
        width, height = image.size

        crop_scale = random.uniform(0.8, 1.0)
        crop_width = int(width * crop_scale)
        crop_height = int(height * crop_scale)

        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)

        crop_box = (
            left,
            top,
            left + crop_width,
            top + crop_height,
        )

        image = image.crop(crop_box)
        label = label.crop(crop_box)

        image = image.resize(
            (width, height),
            resample=Image.Resampling.BILINEAR,
        )

        label = label.resize(
            (width, height),
            resample=Image.Resampling.NEAREST,
        )

    # # Color changes: image only
    # if random.random() < 1.4:
    #     brightness_factor = random.uniform(0.8, 1.2)
    #     image = ImageEnhance.Brightness(image).enhance(
    #         brightness_factor
    #     )

    # if random.random() < 1.4:
    #     contrast_factor = random.uniform(0.8, 1.2)
    #     image = ImageEnhance.Contrast(image).enhance(
    #         contrast_factor
    #     )

    # if random.random() < 0.3:
    #     saturation_factor = random.uniform(0.8, 1.2)
    #     image = ImageEnhance.Color(image).enhance(
    #         saturation_factor
    #     )

    # # Optional mild blur on image only
    # if random.random() < 0.1:
    #     image = image.filter(ImageFilter.GaussianBlur(radius=0.5))

    return image, label


def find_matching_label(label_dir, image_path):
    """
    Match image and label by filename stem.

    Example:
        images/example_001.jpg
        labels/example_001.png
    """

    matches = [
        path
        for path in label_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.stem == image_path.stem
    ]

    if not matches:
        return None

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple labels found for {image_path.name}: {matches}"
        )

    return matches[0]


def save_image(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    # PNG is safest for segmentation labels because it is lossless.
    image.save(path)


def process_split(
    input_split_dir,
    output_split_dir,
    copies_per_image,
    include_originals,
):
    input_image_dir = input_split_dir / "images"
    input_label_dir = input_split_dir / "labels"

    output_image_dir = output_split_dir / "images"
    output_label_dir = output_split_dir / "labels"

    if not input_image_dir.exists():
        print(f"Skipping missing directory: {input_image_dir}")
        return

    if not input_label_dir.exists():
        print(f"Skipping missing directory: {input_label_dir}")
        return

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in input_image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    processed = 0
    skipped = 0

    for image_path in image_paths:
        label_path = find_matching_label(
            input_label_dir,
            image_path,
        )

        if label_path is None:
            print(
                f"WARNING: no matching label for "
                f"{image_path.name}"
            )
            skipped += 1
            continue

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        with Image.open(label_path) as label_file:
            label = label_file.convert("RGB")

        if image.size != label.size:
            raise ValueError(
                f"Image/label size mismatch:\n"
                f"Image: {image_path} - {image.size}\n"
                f"Label: {label_path} - {label.size}"
            )

        width, height = image.size

        if (width, height) != (256, 256):
            print(
                f"WARNING: {image_path.name} has size "
                f"{image.size}, expected (256, 256)"
            )

        # Copy the original files if requested.
        if include_originals:
            save_image(
                image,
                output_image_dir / image_path.name,
            )
            save_image(
                label,
                output_label_dir / label_path.name,
            )

        # Create augmented copies.
        for index in range(copies_per_image):
            augmented_image, augmented_label = (
                build_augmented_pair(image, label)
            )

            image_name = (
                f"{image_path.stem}_aug_{index:03d}.png"
            )
            label_name = (
                f"{label_path.stem}_aug_{index:03d}.png"
            )

            save_image(
                augmented_image,
                output_image_dir / image_name,
            )
            save_image(
                augmented_label,
                output_label_dir / label_name,
            )

        processed += 1

    print(
        f"{input_split_dir.name}: "
        f"processed={processed}, skipped={skipped}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Pillow-based image segmentation augmentation"
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input dataset directory",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output dataset directory",
    )

    parser.add_argument(
        "--copies",
        type=int,
        default=3,
        help="Augmented copies per image",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train"],
        help="Splits to process",
    )

    parser.add_argument(
        "--include-originals",
        action="store_true",
        help="Copy original images and labels too",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results",
    )

    args = parser.parse_args()

    if args.copies < 1:
        raise ValueError("--copies must be at least 1")

    if args.seed is not None:
        random.seed(args.seed)

    for split in args.splits:
        process_split(
            input_split_dir=args.input / split,
            output_split_dir=args.output / split,
            copies_per_image=args.copies,
            include_originals=args.include_originals,
        )


if __name__ == "__main__":
    main()
