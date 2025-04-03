# Automatic detection of pest plants in agriculture using artificial intelligence

## Abstract
The increasing challenges in agriculture - such as labour shortages, environmental pollution from herbicides and the need for sustainable solutions - require new technological approaches to weed control. In this thesis, deep learning models for the automated detection of weeds based on the CropAndWeed dataset are trained and evaluated. The aim is to develop a powerful model for real-time weed detection in resource-limited, autonomous agricultural robots that can compete with comparable models in the literature. For this purpose, three model sizes of the latest YOLOv11 architecture are tested in different training strategies. The results show that even the smallest model outperforms the significantly larger reference model of the dataset authors in terms of recognition performance. In particular, the inclusion of the additional class vegetation enables improved differentiation between weeds and background, which is crucial for real-life application scenarios. Targeted settings in inference mode can further increase decision reliability. The work shows that compact deep learning models in combination with suitable training and inference strategies offer a promising basis for intelligent systems for weed detection and can thus make an important contribution to more sustainable agriculture.
## Usage
1. Copy the CropAndWeed dataset into root directory: https://github.com/cropandweed/cropandweed-dataset
2. Extract the dataset following the instructions in the dataset repository.
3. Install the required packages: ``pip install -r YOLO/requirements.txt``
4. Prepare the dataset mapping you want to use: Run the YOLODataclass (data_preprocessing/dataset_to_yolo/YOLODataclass.py) with the desired dataset mapping as String. Set use_eval=True if you want to use the Eval-Labels with the additional class Vegetation.
5. Modify the Dockerfile: Copy your dataset mapping and the model you want to train. In YOLO/models you can find pretrained models: 
   - cow_default_finetuned_1280_best: Contains the best YOLOv11n, s, m for the CropOrWeed2 mapping trained with imgsz 1280
   - cow_eval_default_finetuned_1280_best: Contains the best YOLOv11n, s, m for the CropOrWeed2Eval mapping with additional Vegetation class trained with imgsz 1280 
   - ...
6. Modify YOLO/scripts/train_eval.py: Set the path to your dataset mapping and the model you want to train.
7. Set the parameters you want to use for training. Have a look at the Ultralytics YOLOv11 documentation for the parameters you can use: https://docs.ultralytics.com/modes/train/
8. Run ``docker compose up`` on the server of your choice and train the models.

## Results
The detailed results and logs of the different training strategies can be found in the runs folder and in the evaluation and discussion of the thesis.