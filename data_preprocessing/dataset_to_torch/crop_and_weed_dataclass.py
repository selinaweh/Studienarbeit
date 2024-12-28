import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from config import IMG_DIR, BBOXES_DIR
from torchvision.io import read_image

class CropAndWeedDataset(Dataset):
    def __init__(self, dataset: str, img_dir, bboxes_dir, labels_dir=None, params_dir=None,
                 transform=None, target_transform=None):
        self.dataset = dataset
        self.img_dir = img_dir
        self.bboxes_dir = os.path.join(bboxes_dir, self.dataset, f"{self.dataset}_annotations.csv")
        self.bboxes = pd.read_csv(self.bboxes_dir)
        #self.labels_dir = labels_dir  # Optional, used only for segmentation tasks
        #self.params_dir = params_dir  # Optional, used for additional parameters
        self.transform = transform
        self.target_transform = target_transform

        #self.dataset_info = DATASETS.get(dataset)
        #self.params = self.load_params() if self.params_dir else {}
        #self.labels = self.load_labels() if self.labels_dir else {}
    '''
    def load_params(self):
        params = {}
        for param_file in os.listdir(self.params_dir):
            if param_file.endswith(".csv"):
                img_id = os.path.splitext(param_file)[0]
                param_path = os.path.join(self.params_dir, param_file)
                param_data = pd.read_csv(param_path)
                params[img_id] = {
                    "moisture": int(param_data["moisture"].iloc[0]),
                    "soil": int(param_data["soil"].iloc[0]),
                    "lighting": int(param_data["lighting"].iloc[0]),
                    "separability": int(param_data["separability"].iloc[0])
                }
        return params

    def load_labels(self):
        labels = {}
        for label_file in os.listdir(self.labels_dir):
            if label_file.endswith(".png"):
                img_id = os.path.splitext(label_file)[0]
                label_path = os.path.join(self.labels_dir, label_file)
                labels[img_id] = label_path
        return labels
    '''
    def __len__(self):
        return len(self.bboxes)

    def __getitem__(self, idx):
        img_id = self.bboxes["Image ID"][idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = read_image(img_path)

        if self.transform:
            image = self.transform(image)

        img_annotations = self.bboxes[self.bboxes["Image ID"] == img_id]
        boxes = img_annotations[["Left", "Top", "Right", "Bottom"]].values
        labels = img_annotations["Label ID"].tolist()

        # optional: map label ids to dataset-specific ids
        if self.dataset_info:
            labels = [self.dataset_info.get_mapped_id(label_id) for label_id in labels]

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        # add additional parameters if available
        if img_id in self.params:
            target.update(self.params[img_id])

        # add segmentation mask if available
        if img_id in self.labels:
            mask = Image.open(self.labels[img_id])
            if self.target_transform:
                mask = self.target_transform(mask)
            target["segmentation_mask"] = mask

        return image, target

    def get_label_color(self, label_id):
        if self.dataset_info:
            return self.dataset_info.get_label_color(label_id)
        return None

    def get_label_name(self, label_id):
        if self.dataset_info:
            return self.dataset_info.get_label_name(label_id)
        return None


if __name__ == "__main__":
    crop_and_weed = CropAndWeedDataset(
        dataset="Soy1",
        img_dir=IMG_DIR,
        bboxes_dir=BBOXES_DIR
    )

    print(crop_and_weed.__len__())
    print(crop_and_weed.__getitem__(0))



    # dataloader = DataLoader(crop_and_weed, batch_size=4, shuffle=True, num_workers=4)
