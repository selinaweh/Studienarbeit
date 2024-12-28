import os
import sys


project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_dir)

# original dataset
DATADICTS = os.path.join(project_dir, "cropandweed_dataset/cnw/utilities/datasets.py")
DATA_DIR = os.path.join(project_dir, "cropandweed_dataset/data/")
IMG_DIR = os.path.join(DATA_DIR, "images")
BBOXES_DIR = os.path.join(DATA_DIR, "bboxes")
LABELS_DIR = os.path.join(DATA_DIR, "labelIds")
PARAMS_DIR = os.path.join(DATA_DIR, "params")

# YOLO datasets
YOLO = os.path.join(project_dir, "YOLO/")
YOLO_DATA_DIR = os.path.join(YOLO, "datasets/")

