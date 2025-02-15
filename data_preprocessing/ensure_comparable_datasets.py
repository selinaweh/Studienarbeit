import os
import json
import shutil
from config import YOLO_DATA_DIR

crop_or_weed2 = os.path.join(YOLO_DATA_DIR, "CropOrWeed2") # reference
coarse1 = os.path.join(YOLO_DATA_DIR, "Coarse1")
crop_or_weed2_eval = os.path.join(YOLO_DATA_DIR, "CropOrWeed2Eval")


def create_empty_label_files(base_path):
    """
    Create empty label files for images that do not have any objects.
    """
    splits = ["train", "val", "test"]

    for split in splits:
        img_dir = os.path.join(base_path, "images", split)
        label_dir = os.path.join(base_path, "labels", split)

        os.makedirs(label_dir, exist_ok=True)

        for img_file in os.listdir(img_dir):
            if img_file.endswith('.jpg'):
                img_name = os.path.splitext(img_file)[0]
                label_file = os.path.join(label_dir, img_name + ".txt")

                if not os.path.exists(label_file):
                    with open(label_file, "w") as f:
                        pass
                    print(f"Created: {label_file}")


""" 
    Dataset Variants were created by random split, 
    so the train results are not comparable (data distribution is different).
    This script is used to apply the same distribution to all datasets after creation with YOLODataclass.
"""


def extract_distribution(base_path):
    """
    Extract the image-label distribution from a dataset structure.
    """
    distribution = {}
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(base_path, "images", split)
        images = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))]
        distribution[split] = images

    with open("split_distribution.json", "w") as f:
        json.dump(distribution, f)

    return distribution


def apply_distribution(base_path, distribution):
    """
    Apply a given distribution to a dataset by reorganizing images and labels,
    only moving files that are in the wrong split folder.

    Args:
        base_path (str): Path to the dataset base directory.
        distribution (dict): Dictionary with splits as keys and image names (without extensions) as values.
    """
    image_to_split = {img_name: split for split, images in distribution.items() for img_name in images}

    for split, images in distribution.items():
        img_dir = os.path.join(base_path, "images", split)
        label_dir = os.path.join(base_path, "labels", split)

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        for current_split in ["train", "val", "test"]:
            current_img_dir = os.path.join(base_path, "images", current_split)
            current_label_dir = os.path.join(base_path, "labels", current_split)

            current_images = [os.path.splitext(f)[0] for f in os.listdir(current_img_dir) if f.endswith((".jpg", ".png"))]

            for img_name in current_images:
                target_split = image_to_split.get(img_name)

                if target_split != current_split:
                    src_img_file = os.path.join(current_img_dir, img_name + ".jpg")
                    dst_img_file = os.path.join(base_path, "images", target_split, img_name + ".jpg")
                    if os.path.exists(src_img_file):
                        shutil.move(src_img_file, dst_img_file)
                        print(f"Moved image: {src_img_file} -> {dst_img_file}")

                    src_label_file = os.path.join(current_label_dir, img_name + ".txt")
                    dst_label_file = os.path.join(base_path, "labels", target_split, img_name + ".txt")
                    if os.path.exists(src_label_file):
                        shutil.move(src_label_file, dst_label_file)
                        print(f"Moved label: {src_label_file} -> {dst_label_file}")



def verify_split_distribution(base_path, distribution_file):
    """
    Verifies that the image and label distribution in a dataset matches the given distribution.

    Args:
        base_path (str): Path to the dataset directory to verify.
        distribution_file (str): Path to the JSON file containing the reference distribution.

    Returns:
        bool: True if the distribution matches, False otherwise.
    """
    # Load the reference distribution from the JSON file
    with open(distribution_file, "r") as f:
        reference_distribution = json.load(f)

    all_match = True

    # Check each split
    for split, reference_images in reference_distribution.items():
        img_dir = os.path.join(base_path, "images", split)
        label_dir = os.path.join(base_path, "labels", split)

        dataset_images = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith((".jpg"))]
        dataset_labels = [os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith(".txt")]

        missing_images = set(reference_images) - set(dataset_images)
        extra_images = set(dataset_images) - set(reference_images)

        missing_labels = set(reference_images) - set(dataset_labels)

        if missing_images:
            print(f"Missing images in '{split}' split: {missing_images}")
            all_match = False

        if extra_images:
            print(f"Extra images in '{split}' split: {extra_images}")
            all_match = False

        if missing_labels:
            print(f"Missing labels in '{split}' split: {missing_labels}")
            all_match = False

        if not missing_images and not extra_images and not missing_labels:
            print(f"'{split}' split matches the reference distribution.")

    return all_match


if __name__ == "__main__":

    with open("split_distribution.json", "r") as f:
        dist = json.load(f)

    apply_distribution(crop_or_weed2_eval, dist)

    if verify_split_distribution(crop_or_weed2_eval, "split_distribution.json"):
        print("The dataset matches the reference distribution.")
    else:
        print("The dataset does not match the reference distribution.")
