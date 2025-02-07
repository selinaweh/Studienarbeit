local = False
'''
if local:
    from config import YOLO_DATA_DIR, YOLO_DIR
    import os

    # datasets
    DATASETS_DIR = YOLO_DATA_DIR
    # dataset yaml paths
    FINE24 = os.path.join(YOLO_DATA_DIR, "Fine24/Fine24_local.yaml")
    CROP_OR_WEED2 = os.path.join(YOLO_DATA_DIR, "CropOrWeed2/CropOrWeed2_local.yaml")

    # model paths
    MODEL_PRETRAINED = os.path.join(YOLO_DIR, "models_trained/yolo11n.pt")

    # save paths
    YOLO_FINE24 = os.path.join(YOLO_DIR, "models_trained/yolo11n_fine24.pt")
    YOLO_CROP_OR_WEED2 = os.path.join(YOLO_DIR, "models_trained/yolo11n_crop_or_weed2.pt")
    YOLO_BOTH = os.path.join(YOLO_DIR, "models_trained/yolo11n_both.pt")

else:
    # datasets
    DATASETS_DIR = "/app/datasets/"

    # dataset yaml paths
    CROP_OR_WEED2 = "/app/datasets/CropOrWeed2/CropOrWeed2.yaml"
    FINE24 = "/app/datasets/Fine24/Fine24.yaml"

    # model paths
    MODEL_PRETRAINED = "/app/models_trained/yolo11n.pt"

    # save paths
    YOLO_FINE24 = "/app/models_trained/yolo11n_fine24.pt"
    YOLO_CROP_OR_WEED2 = "/app/models_trained/yolo11n_crop_or_weed2.pt"
    YOLO_BOTH = "/app/models_trained/yolo11n_both.pt"
'''