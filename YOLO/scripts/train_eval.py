from ultralytics import YOLO, settings
import os
import torch
#from ..config_paths import (DATASETS_DIR, FINE24, CROP_OR_WEED2, MODEL_PRETRAINED, YOLO_CROP_OR_WEED2, YOLO_FINE24, YOLO_BOTH)
#from config import YOLO_DATA_DIR, YOLO_DIR
import yaml



#fine24 = os.path.join(YOLO_DATA_DIR, "Fine24/Fine24.yaml")
#crop_or_weed2 = os.path.join(YOLO_DATA_DIR, "CropOrWeed2/CropOrWeed2.yaml")
crop_or_weed2 = "/app/datasets/CropOrWeed2/CropOrWeed2.yaml"
model_pretrained = "/app/models/yolo11n.pt"
#model_pretrained = os.path.join(YOLO_DIR, "models/yolo11n.pt")

#yolo_fine24 = os.path.join(YOLO_DIR, "models/yolo11n_fine24.pt")
#yolo_crop_or_weed2 = os.path.join(YOLO_DIR, "models/yolo11n_crop_or_weed2.pt")
#yolo_both = os.path.join(YOLO_DIR, "models/yolo11n_both.pt")

yolo_fine24 = "/app/models/yolo11n_fine24.pt"
yolo_crop_or_weed2 = "/app/models/yolo11n_crop_or_weed2.pt"
yolo_both = "/app/models/yolo11n_both.pt"


def update_settings():
    settings.update({
        "datasets_dir": "/app/datasets/", #YOLO_DATA_DIR,
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


def get_latest_tune_dir(base_dir="runs/detect"):
    if not os.path.exists(base_dir):
        print(f"Base directory {base_dir} does not exist.")
        return None

    tune_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
                 if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("tune")]

    if not tune_dirs:
        print(f"No tuning directories found in {base_dir}.")
        return None

    latest_tune_dir = max(tune_dirs, key=os.path.getmtime)
    return latest_tune_dir


def train_and_validate_model(
    model,
    dataset,
    device,
    save_path,
    enable_tuning=False,
    tune_epochs=2,
    tune_iterations=6,
    train_epochs=5,
    batch=-1,
    optimizer="auto",
    patience=10,
    tune_optimizer="AdamW",
    tune_plots=True,
    tune_save=True,
    tune_val=True
):
    """
    Trains, validates, and saves a model with optional hyperparameter tuning.

    Parameters:
        model: The model to be trained and validated.
        dataset: The dataset to be used for training and validation.
        device: The device (e.g., 'cpu', 'cuda') to be used.
        save_path: Path to save the trained model.
        enable_tuning: Whether to perform hyperparameter tuning (default: False).
        tune_epochs: Number of epochs for hyperparameter tuning.
        tune_iterations: Number of iterations for hyperparameter tuning.
        train_epochs: Number of epochs for training.
        batch: Batch size for training and validation (-1 for auto).
        optimizer: Optimizer for training.
        patience: Early stopping patience during training.
        tune_optimizer: Optimizer for hyperparameter tuning.
        tune_plots: Whether to generate plots during hyperparameter tuning.
        tune_save: Whether to save hyperparameter tuning results.
        tune_val: Whether to perform validation during tuning.
    """
    hyperparameters = {}

    # Hyperparameter tuning (optional)
    if enable_tuning:
        try:
            model.tune(
                data=dataset,
                epochs=tune_epochs,
                iterations=tune_iterations,
                optimizer=tune_optimizer,
                plots=tune_plots,
                save=tune_save,
                val=tune_val,
                exist_ok=True
            )
            # Get the best hyperparameters
            latest_tune_dir = get_latest_tune_dir("runs/detect")
            if latest_tune_dir:
                hyp_path = os.path.join(latest_tune_dir, "best_hyperparameters.yaml")
                if os.path.exists(hyp_path):
                    with open(hyp_path, 'r') as file:
                        hyperparameters = yaml.safe_load(file)
        except Exception as e:
            print(f"Error during tuning: {e}")
            print("Proceeding with default hyperparameters.")

    if not hyperparameters:
        print("Using default hyperparameters.")

    # Train the model
    try:
        model.train(
            data=dataset,
            epochs=train_epochs,
            device=device,
            optimizer=optimizer,
            batch=batch,
            patience=patience,
            exist_ok=True,
            **hyperparameters
        )
    except Exception as e:
        print(f"Error during training: {e}")

    # Validation
    try:
        model.val(data=dataset, batch=batch, exist_ok=True)
    except Exception as e:
        print(f"Error during validation: {e}")

    # Save the model
    try:
        print(f"Saving model to: {save_path}")
        model.save(save_path)
    except Exception as e:
        print(f"Error during saving: {e}")


def main():
    # Select device
    device = get_device()
    print(f"Using device: {device}")

    # Update settings
    update_settings()
    print(settings)

    # Load model and dataset
    model_crop_and_weed2 = YOLO(model_pretrained, task='detect')
    #dataset = crop_or_weed2

    train_and_validate_model(model_crop_and_weed2, crop_or_weed2, device, yolo_crop_or_weed2, enable_tuning=True, tune_epochs=10, tune_iterations=300, train_epochs=200)
    '''
    # Hyperparameter tuning
    try:
        model.tune(data=dataset, epochs=2, iterations=6, optimizer="AdamW", plots=True, save=True, val=True, exist_ok=True)
    except Exception as e:
        print(f"Error during tuning: {e}")
        return
    #hyp_path = "runs/detect/tune/best_hyperparameters.yaml"
    latest_tune_dir = get_latest_tune_dir("runs/detect")
    if latest_tune_dir:
        hyp_path = os.path.join(latest_tune_dir, "best_hyperparameters.yaml")
    else:
        hyp_path = None

    if hyp_path and os.path.exists(hyp_path):
        with open(hyp_path, 'r') as file:
            hyperparameters = yaml.safe_load(file)
    else:
        print("Using default hyperparameters.")
        hyperparameters = {}

    # Train the model
    try:
        model.train(data=dataset, epochs=5, device=device, optimizer="auto", batch=-1, patience=10, exist_ok=True, **hyperparameters)
    except Exception as e:
        print(f"Error during training: {e}")

    # Validation
    try:
        model.val(data=dataset, batch=-1, exist_ok=True)
    except Exception as e:
        print(f"Error during validation: {e}")

    # Save the model
    try:
        print(f"Saving model to: {yolo_crop_or_weed2}")
        model.save(yolo_crop_or_weed2)
    except Exception as e:
        print(f"Error during saving: {e}")

    # CropOrWeed2 dataset
    #tune_model(model_pretrained, crop_or_weed2)
    train_model(model_pretrained, crop_or_weed2, yolo_crop_or_weed2, device)#, hyperparams=True)
    evaluate_model(yolo_crop_or_weed2, crop_or_weed2, device)


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
