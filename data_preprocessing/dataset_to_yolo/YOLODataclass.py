import os
import cv2
import pandas as pd
import random
import yaml

from config import IMG_DIR, LABELS_DIR, YOLO_DATA_DIR
from cropandweed_dataset.cnw.utilities.datasets import DATASETS

def split_train_val_test(pairs, train_ratio=0.7, val_ratio=0.15):
    """Train-val-test split 70:15:15 for image-label pairs (analog to CropAndWeed dataset paper)."""
    random.seed(42)
    random.shuffle(pairs)

    total = len(pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    return train_pairs, val_pairs, test_pairs

def convert_labels_to_yolo(label_file, output_file, img_width, img_height, next_free_class_id=0):
    """
    Converts annotations from a CSV file to YOLO format and writes to a .txt file.
    Each line in the output file corresponds to one object in the format:
    class_id x_center y_center width height
    """
    column_names = ["Left", "Top", "Right", "Bottom", "Label ID", "Stem X", "Stem Y"]
    df = pd.read_csv(label_file, header=None, names=column_names)
    yolo_annotations = []

    for _, row in df.iterrows():
        left, top, right, bottom = row["Left"], row["Top"], row["Right"], row["Bottom"]
        class_id = int(row["Label ID"])  # Use class numbers directly (zero-indexed)

        # Replace 255 with the next available class ID
        if class_id == 255:
            class_id = next_free_class_id

        # Convert bboxes to YOLO format (normalized values)
        x_center = (left + right) / 2 / img_width
        y_center = (top + bottom) / 2 / img_height
        width = (right - left) / img_width
        height = (bottom - top) / img_height

        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Write to output .txt file
    with open(output_file, "w") as f:
        if yolo_annotations:
            f.write("\n".join(yolo_annotations))
        else:
            # Create an empty file if no annotations are present
            f.write("")

class YOLOCropAndWeed:
    def __init__(self, dataset: str, src_img_dir, src_labels_dir, yolo_data_dir, use_eval=False):
        self.dataset = dataset
        self.dataset_info = DATASETS.get(self.dataset)
        self.class_names = {key: value[0] for key, value in self.dataset_info.labels.items()} if self.dataset_info else {}
        self.use_eval = use_eval

        # Adjust destination directories based on use_eval flag
        if self.use_eval:
            self.dataset_suffix = f"{self.dataset}Eval"
            self.vegetation_class_id = max(self.class_names.keys(), default=-1) + 1
        else:
            self.dataset_suffix = self.dataset

        self.yolo_data_dir = yolo_data_dir
        self.src_img_dir = src_img_dir
        self.src_labels_dir = os.path.join(src_labels_dir, f"{self.dataset}Eval" if self.use_eval else self.dataset)
        self.dst_img_dir = os.path.join(yolo_data_dir, self.dataset_suffix, "images")
        self.dst_labels_dir = os.path.join(yolo_data_dir, self.dataset_suffix, "labels")

        if not os.path.exists(self.yolo_data_dir):
            os.mkdir(self.yolo_data_dir)


    def make_data_dirs(self):
        for base_dir in [self.dst_img_dir, self.dst_labels_dir]:
            os.makedirs(base_dir, exist_ok=True)
            for sub_dir in ["train", "val", "test"]:
                os.makedirs(os.path.join(base_dir, sub_dir), exist_ok=True)

    def map_images_to_labels(self):
        images = [f for f in os.listdir(self.src_img_dir)]
        labels = [f for f in os.listdir(self.src_labels_dir)]

        image_label_pairs = []
        for img in images:
            label = os.path.splitext(img)[0] + ".csv"
            if label in labels:
                image_label_pairs.append((img, label))
            else:
                image_label_pairs.append((img, None))  # Include images without labels
        return image_label_pairs

    def create_yaml_file(self):
        """
        Create a YAML file for the dataset containing the paths to the images and labels.
        """
        # Add Vegetation class if use_eval is True
        if self.use_eval:
            self.class_names[self.vegetation_class_id] = "Vegetation"

        yaml_data = {
            "path": f"/app/datasets/{self.dataset_suffix}",
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": self.class_names
        }

        yaml_file_path = os.path.join(self.yolo_data_dir, self.dataset_suffix, f"{self.dataset_suffix}.yaml")
        with open(yaml_file_path, "w") as yaml_file:
            yaml.dump(yaml_data, yaml_file, default_flow_style=False, sort_keys=False)

    def prepare_dataset(self):

        self.make_data_dirs()
        self.create_yaml_file()
        pairs = self.map_images_to_labels()
        print(f"Found {len(pairs)} image-label pairs for dataset '{self.dataset}'.")
        train_pairs, val_pairs, test_pairs = split_train_val_test(pairs)

        for split, split_pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
            for img_file, label_file in split_pairs:
                img_src = os.path.join(self.src_img_dir, img_file)

                img_dst = os.path.join(self.dst_img_dir, split, img_file)
                label_dst = os.path.join(self.dst_labels_dir, split, os.path.splitext(img_file)[0] + ".txt")

                # Save images directly without normalization (not necessary for YOLO)
                img = cv2.imread(img_src)
                if img is None:
                    raise FileNotFoundError(f"Could not find file: {img_src}")
                cv2.imwrite(img_dst, img)

                # Convert labels to YOLO format
                if label_file:
                    label_src = os.path.join(self.src_labels_dir, label_file)
                    img_height, img_width = img.shape[:2]
                    convert_labels_to_yolo(label_src, label_dst, img_width, img_height, self.vegetation_class_id)
                else:
                    # Create an empty .txt file for images without labels
                    open(label_dst, 'w').close()

if __name__ == "__main__":
    # Coarse1 distinguishes only between Vegetation 0 and Background (only Vegetation class 0)
    # Coarse1Eval includes label files for images without labels as empty files
    yolo_coarse1 = YOLOCropAndWeed("Coarse1", IMG_DIR, LABELS_DIR, YOLO_DATA_DIR, use_eval=True)
    yolo_coarse1.prepare_dataset()

    # CropOrWeed2 distinguishes only between crop and weed
    #yolo_crop_or_weed = YOLOCropAndWeed("CropOrWeed2", IMG_DIR, BBOXES_DIR, YOLO_DATA_DIR, use_eval=True)
    #yolo_crop_or_weed.prepare_dataset()
