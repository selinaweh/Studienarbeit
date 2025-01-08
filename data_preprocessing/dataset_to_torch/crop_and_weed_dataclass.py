import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from config import IMG_DIR, BBOXES_DIR
from torchvision import transforms as T
from torchvision.io import read_image


def convert_labels_to_pt(label_file):
    """
    Converts annotations from a CSV file to PyTorch format.

    Parameters:
    - label_file (str): Path to the input CSV file with annotations.

    Returns:
    - targets (list[dict]): List of dictionaries, each containing:
        - boxes (FloatTensor[N, 4]): Ground-truth boxes in [x1, y1, x2, y2] format.
        - labels (Int64Tensor[N]): Class labels for each ground-truth box.
    """
    column_names = ["Left", "Top", "Right", "Bottom", "Label ID", "Stem X", "Stem Y"]
    df = pd.read_csv(label_file, header=None, names=column_names)

    # Create the boxes and labels tensors
    boxes = []
    labels = []

    for _, row in df.iterrows():
        left, top, right, bottom = row["Left"], row["Top"], row["Right"], row["Bottom"]
        class_id = row["Label ID"]
        # Append box in [x1, y1, x2, y2] format
        boxes.append([left, top, right, bottom])
        labels.append(class_id)

    # Convert to tensors
    boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
    labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

    # Create the target dictionary for a single image
    target = {
        "boxes": boxes_tensor,
        "labels": labels_tensor
    }
    return target


class CropAndWeedDataset(Dataset):

    def __init__(self, dataset: str, src_img_dir, src_labels_dir, pt_data_dir, transform=None, target_transform=None):
        self.dataset = dataset
        self.pt_data_dir = pt_data_dir
        self.src_img_dir = src_img_dir
        self.src_labels_dir = os.path.join(src_labels_dir, self.dataset)
        self.dst_img_dir = os.path.join(pt_data_dir, dataset, "images")
        self.dst_labels_dir = os.path.join(pt_data_dir, dataset, "labels")

        if not os.path.exists(self.pt_data_dir):
            os.mkdir(self.pt_data_dir)

        self.transform = transform
        self.target_transform = target_transform

        self.image_label_pairs = self.map_images_to_labels()

    def map_images_to_labels(self):
        images = [f for f in os.listdir(self.src_img_dir)]
        labels = [f for f in os.listdir(self.src_labels_dir)]

        image_label_pairs = []
        for img in images:
            label = os.path.splitext(img)[0] + ".csv"
            if label in labels:
                image_label_pairs.append((img, label))
        return image_label_pairs

    def __len__(self):
        return len(os.listdir(self.image_label_pairs))

    def __getitem__(self, idx):
        img_name, label_name = self.image_label_pairs[idx]

        # Load image
        img_path = os.path.join(self.src_img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image /= 255.0
        image = torch.as_tensor(image)

        # Load labels
        label_path = os.path.join(self.src_labels_dir, label_name)
        target = convert_labels_to_pt(label_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target


def get_transform():
    """
    The transform pipeline makes it possible to perform operations such as
    normalization, data augmentation, and other preprocessing steps that can
    help to improve the performance of a neural network during training.
    """
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


if __name__ == "__main__":
    crop_and_weed = CropAndWeedDataset(
        dataset="Soy1",
        img_dir=IMG_DIR,
        bboxes_dir=BBOXES_DIR
    )

    print(crop_and_weed.__len__())
    print(crop_and_weed.__getitem__(0))



    # dataloader = DataLoader(crop_and_weed, batch_size=4, shuffle=True, num_workers=4)
