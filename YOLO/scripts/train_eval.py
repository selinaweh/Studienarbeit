from ultralytics import YOLO, settings
import os
import torch
import time
import yaml

#_________________________datasets__________________________

# Dataset crop_or_weed2 uses normal labels without 255:Vegetation class
# with an empty .txt file for images without objects (added manually)
crop_or_weed2 = "/app/datasets/CropOrWeed2/CropOrWeed2.yaml"


# Dataset crop_or_weed2_eval uses Eval labels with 255:Vegetation class converted to 2:Vegetation
# with an empty .txt file for images without objects (already included in the dataset)
crop_or_weed2_eval = "/app/datasets/CropOrWeed2Eval/CropOrWeed2Eval.yaml"

# Coarse1 distinguishes only between Vegetation 0 and Background (only Vegetation class 0)
# with an empty .txt file for images without objects (added manually, because Dataset was
# already created; the only difference Coarse1 and Coarse1Eval are the empty .txt files,
# no additional entries for this dataset subvariant)
coarse1 = "/app/datasets/Coarse1/Coarse1.yaml"


#____________________models pretrained________________________
model_yolo11n = "/app/models/yolo11n.pt" # nano Yolo11n
model_yolo11s = "/app/models/yolo11s.pt" # small Yolo11s
model_yolo11m = "/app/models/yolo11m.pt" # medium Yolo11m




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
    imgsz=640,
    enable_tuning=False,
    tune_epochs=10,
    tune_iterations=100,
    train_epochs=100,
    batch=16,
    optimizer="auto",
    patience=10,
    hyperparameters=None,
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
        hyperparameters: Dictionary of hyperparameters for training.
        tune_optimizer: Optimizer for hyperparameter tuning.
        tune_plots: Whether to generate plots during hyperparameter tuning.
        tune_save: Whether to save hyperparameter tuning results.
        tune_val: Whether to perform validation during tuning.
    """

    # Hyperparameter tuning (optional)
    if hyperparameters is None:
        hyperparameters = {}
    if enable_tuning:
        try:
            model.tune(
                data=dataset,
                epochs=tune_epochs,
                iterations=tune_iterations,
                optimizer=tune_optimizer,
                imgsz=imgsz,
                device=device,
                batch=batch,
                plots=tune_plots,
                save=tune_save,
                val=tune_val,
                exist_ok=False,
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
            imgsz=imgsz,
            optimizer=optimizer,
            batch=batch,
            patience=patience,
            exist_ok=False,
            **hyperparameters
        )
    except Exception as e:
        print(f"Error during training: {e}")

    # Val on validation set
    try:
        model.val(data=dataset, batch=batch, exist_ok=False)
    except Exception as e:
        print(f"Error during validation: {e}")

    # Val on test set
    try:
        model.val(data=dataset, batch=batch, split='test', exist_ok=False)
    except Exception as e:
        print(f"Error during testing: {e}")


def main():
    # Select device
    device = get_device()
    print(f"Using device: {device}")

    # Update settings
    update_settings()
    print(settings)

    #________________Load model and dataset for training_____________________

    # Load model and dataset: YOLO11n with default params and imgsz640 and CropOrWeed2, train 50 epochs
    yolo11n = YOLO("/app/models/yolo11n.pt", task='detect')
    train_and_validate_model(yolo11n, crop_or_weed2, device, train_epochs=50, imgsz=640)
    '''
    torch.cuda.empty_cache()

    # Load model and dataset: YOLO11s with default params and imgsz1280 and Coarse1, train 50 epochs
    yolo11s_default_best = YOLO("/app/models/cow_default_best/cow_yolo11s_default_best.pt", task='detect')
    train_and_validate_model(yolo11s_default_best, crop_or_weed2, device, train_epochs=50, imgsz=1920)

    torch.cuda.empty_cache()

    # Load model and dataset: YOLO11m with default params and imgsz1280 and Coarse1, train 50 epochs
    yolo11m_default_best = YOLO("/app/models/cow_default_best/cow_yolo11m_default_best.pt", task='detect')
    train_and_validate_model(yolo11m_default_best, crop_or_weed2, device, train_epochs=50, imgsz=1920)

    torch.cuda.empty_cache()

    yolo11l_default = YOLO("/app/models/yolo11l.pt", task='detect')
    train_and_validate_model(yolo11l_default, crop_or_weed2, device, train_epochs=100, imgsz=1280)

    # Load model and dataset: YOLO11s and CropOrWeed2
    cow_yolo11s_default_finetuned_imgsz_1280 = YOLO("/app/models/cow_default_finetuned_1280_best/cow_yolo11s_default_finetuned_imgsz_1280.pt", task='detect')
    train_and_validate_model(cow_yolo11s_default_finetuned_imgsz_1280, crop_or_weed2, device, enable_tuning=True, train_epochs=50, imgsz=1280)



    # Load model and dataset: YOLO11m and CropOrWeed2
    cow_yolo11m_default = YOLO("/app/models/cow_default_best/cow_yolo11m_default_best.pt", task='detect')
    cow_yolo11m_default.val(data=crop_or_weed2, split='test', exist_ok=False)
    train_and_validate_model(cow_yolo11m_default, crop_or_weed2, device, train_epochs=50, imgsz=1280)

    torch.cuda.empty_cache()
    #___________________________________________________________________________________________

    # Load model and dataset: YOLO11n and CropOrWeed2Eval
    cow_eval_yolo11n_default = YOLO("/app/models/cow_eval_default_best/cow_eval_yolo11n_default_best.pt", task='detect')
    cow_eval_yolo11n_default.val(data=crop_or_weed2_eval, split='test', exist_ok=False)
    train_and_validate_model(cow_eval_yolo11n_default, crop_or_weed2_eval, device, train_epochs=50, imgsz=1280)

    torch.cuda.empty_cache()

    # Load model and dataset: YOLO11s and CropOrWeed2Eval
    cow_eval_yolo11s_default = YOLO("/app/models/cow_eval_default_best/cow_eval_yolo11s_default_best.pt", task='detect')
    cow_eval_yolo11s_default.val(data=crop_or_weed2_eval, split='test', exist_ok=False)
    train_and_validate_model(cow_eval_yolo11s_default, crop_or_weed2_eval, device, train_epochs=50, imgsz=1280)

    torch.cuda.empty_cache()

    # Load model and dataset: YOLO11m and CropOrWeed2Eval
    cow_eval_yolo11m_default = YOLO("/app/models/cow_eval_default_best/cow_eval_yolo11m_default_best.pt", task='detect')
    cow_eval_yolo11m_default.val(data=crop_or_weed2_eval, split='test', exist_ok=False)
    train_and_validate_model(cow_eval_yolo11m_default, crop_or_weed2_eval, device, train_epochs=50, imgsz=1280)
    '''

    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
