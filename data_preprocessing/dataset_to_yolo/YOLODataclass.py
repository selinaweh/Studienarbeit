import os
import cv2
import pandas as pd
import random
import yaml

from config import IMG_DIR, BBOXES_DIR, YOLO_DATA_DIR
from cropandweed_dataset.cnw.utilities.datasets import DATASETS


def split_train_val_test(pairs, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Train-val-test split 70:20:10 for image-label pairs as recommended in the ultralytics documentation."""
    # Shuffle pairs for random splitting
    random.shuffle(pairs)

    total = len(pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    # Split YOLO
    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    return train_pairs, val_pairs, test_pairs


def convert_labels_to_yolo(label_file, output_file, img_width, img_height):
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
        class_id = row["Label ID"]

        # convert bboxes to YOLO format
        x_center = (left + right) / (2 * img_width)
        y_center = (top + bottom) / (2 * img_height)
        width = (right - left) / img_width
        height = (bottom - top) / img_height

        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # wirte to output .txt file
    with open(output_file, "w") as f:
        f.write("\n".join(yolo_annotations))


def resize_and_save_images(src_file, dst_file, size=(640, 640)):
    img = cv2.imread(src_file)
    if img is None:
        raise FileNotFoundError(f"Could not find file: {src_file}")
    img_height, img_width = img.shape[:2]
    resized_img = cv2.resize(img, size)
    cv2.imwrite(dst_file, resized_img)
    return img_width, img_height


class YOLOCropAndWeed:
    def __init__(self, dataset: str, src_img_dir, src_labels_dir, yolo_data_dir):
        self.dataset = dataset
        self.yolo_data_dir = yolo_data_dir
        self.src_img_dir = src_img_dir
        self.src_labels_dir = os.path.join(src_labels_dir, self.dataset)
        self.dst_img_dir = os.path.join(yolo_data_dir, dataset, "images")
        self.dst_labels_dir = os.path.join(yolo_data_dir, dataset, "labels")

        if not os.path.exists(self.yolo_data_dir):
            os.mkdir(self.yolo_data_dir)

        self.make_data_dirs()
        self.prepare_dataset()
        self.create_yaml_file()

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
        return image_label_pairs

    def create_yaml_file(self):
        """
        Create a YAML file for the dataset containing the paths to the images and labels.
        """
        dataset_info = DATASETS.get(self.dataset)
        if not dataset_info:
            raise ValueError(f"Dataset '{self.dataset}' not found in DATASETS.")

        class_names = {key: value[0] for key, value in dataset_info.labels.items()}

        # define YAML YOLO
        yaml_data = {
            "path": f"YOLO/datasets/{self.dataset}",  # YOLO data directory
            "train": "images/train",  # train images
            "val": "images/val",      # validation images
            "test": "images/test",    # test
            "names": class_names      # class names
        }

        yaml_file_path = os.path.join(self.yolo_data_dir, self.dataset, f"{self.dataset}.yaml")
        with open(yaml_file_path, "w") as yaml_file:
            yaml.dump(yaml_data, yaml_file, default_flow_style=False, sort_keys=False)


    def prepare_dataset(self):

        pairs = self.map_images_to_labels()
        train_pairs, val_pairs, test_pairs = split_train_val_test(pairs)

        for split, split_pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
            for img_file, label_file in split_pairs:
                img_src = os.path.join(self.src_img_dir, img_file)
                label_src = os.path.join(self.src_labels_dir, label_file)

                img_dst = os.path.join(self.dst_img_dir, split, img_file)
                label_dst = os.path.join(self.dst_labels_dir, split, os.path.splitext(img_file)[0] + ".txt")

                # scale and save images
                img_width, img_height = resize_and_save_images(img_src, img_dst)

                # convert labels to YOLO format
                convert_labels_to_yolo(label_src, label_dst, img_width, img_height)



if __name__ == "__main__":
    # most fine-grained variant Fine24 maps the original labels
    # into 8 crop and 16 weed classes based primarily on botanical
    # categorization and, in rare cases, visual similarity
    yolo_fine24 = YOLOCropAndWeed("Fine24", IMG_DIR, BBOXES_DIR, YOLO_DATA_DIR)
    yolo_fine24.prepare_dataset()

    # CropOrWeed2 distinguishes only between crop and weed
    yolo_crop_or_weed = YOLOCropAndWeed("CropOrWeed2", IMG_DIR, BBOXES_DIR, YOLO_DATA_DIR)
    yolo_crop_or_weed.prepare_dataset()
