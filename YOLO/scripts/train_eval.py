from types import SimpleNamespace

from ultralytics import YOLO, settings
from ultralytics.cfg import get_save_dir
import os
import sys
import torch
#from config import YOLO_DATA_DIR
#import yaml

#fine24 = os.path.join(YOLO_DATA_DIR, "Fine24/Fine24.yaml")
#crop_or_weed2 = os.path.join(YOLO_DATA_DIR, "CropOrWeed2/CropOrWeed2.yaml")
crop_or_weed2 = "app/datasets/CropOrWeed2/CropOrWeed2.yaml"
model_pretrained = "models/yolo11n.pt"

yolo_fine24 = "models/yolo11n_fine24.pt"
yolo_crop_or_weed2 = "models/yolo11n_crop_or_weed2.pt"
yolo_both = "models/yolo11n_both.pt"


def update_settings():
    settings.update({
        "datasets_dir": "app/datasets/", #YOLO_DATA_DIR,
        "tensorboard": True
    })


def get_device():
    """Select device for training."""
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

'''
def tune_model(model_path, dataset_path, epochs=10, iterations=300, optimizer="Adam"):
    print("Starting hyperparameter tuning...")
    model = YOLO(model_path, task="detect", )
    result_grid = model.tune(
        data=dataset_path,
        epochs=epochs,
        iterations=iterations,
        optimizer=optimizer,
        use_ray=True,
    )


def train_model(model_path, dataset_path, save_path, device, hyperparams=None, optimizer="auto", epochs=300, batch_size=16, patience=10):
    print("Starting training...")
    model = YOLO(model_path, task="detect")
    results = model.train(
        data=dataset_path,
        epochs=epochs,
        hyp='runs/detect/tune/best_hyperparameters.yaml',
        device=device,
        optimizer=optimizer,
        batch_size=batch_size,
        patience=patience,
    )
    model.save(save_path)
    return results
'''
def tune_model(
    model_path,
    dataset_path,
    task="detect",
    epochs=10,
    iterations=300,
    optimizer="Adam",
    use_ray=True
):
    print(f"Starting hyperparameter tuning for task: {task}...")
    model = YOLO(model_path, task=task)
    model.tune(
        data=dataset_path,
        epochs=epochs,
        iterations=iterations,
        optimizer=optimizer,
        use_ray=use_ray
    )
    args = SimpleNamespace(project="YOLO", task="detect", mode="train", exist_ok=True)
    save_dir = get_save_dir(args)
    best_hyperparams_path = f"{save_dir}/best_hyperparameters.yaml"
    print(f"Tuning completed. Best hyperparameters saved at: {best_hyperparams_path}")
    return best_hyperparams_path

def train_model(
    model_path,
    dataset_path,
    save_path,
    device,
    hyperparams_path=None,
    task="detect",
    optimizer="auto",
    epochs=300,
    batch=16,
    patience=10,
):

    print(f"Starting training for task: {task}...")
    model = YOLO(model_path, task=task)

    train_args = {
        "data": dataset_path,
        "epochs": epochs,
        "device": device,
        "optimizer": optimizer,
        "batch": batch,
        "patience": patience,
    }

    if hyperparams_path:
        train_args["hyp"] = hyperparams_path

    results = model.train(**train_args)
    model.save(save_path)
    print(f"Training completed. Model saved at: {save_path}")
    return results


def evaluate_model(model_path, dataset_path, device):
    print("Starting evaluation...")
    model = YOLO(model_path, task="detect")
    metrics = model.val(data=dataset_path, device=device)
    print("Evaluation completed:", metrics)
    return metrics


def main():
    # Select device
    device = get_device()
    print(f"Using device: {device}")

    # Update settings
    update_settings()
    print(settings)

    # CropOrWeed2 dataset
    #best_hp_crop_or_weed2 = tune_model(model_pretrained, crop_or_weed2)
    train_model(model_pretrained, crop_or_weed2, yolo_crop_or_weed2, device)#, best_hp_crop_or_weed2)
    evaluate_model(yolo_crop_or_weed2, crop_or_weed2, device)

    '''
    # Fine24 dataset
    best_hp_fine24 = tune_model(model_pretrained, fine24)
    train_model(model_pretrained, fine24, yolo_fine24, device, best_hp_fine24)
    evaluate_model(yolo_fine24, fine24, device)

    # Train pretrained CropOrWeed2 YOLO on Fine24
    best_hp_both = tune_model(yolo_crop_or_weed2, fine24)
    train_model(yolo_crop_or_weed2, fine24, yolo_both, device, best_hp_both)
    evaluate_model(yolo_both, fine24, device)
    '''

if __name__ == "__main__":
    main()
