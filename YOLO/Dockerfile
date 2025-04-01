FROM python:3.11.0-slim

RUN apt-get update && apt-get install -y \
    wget \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && apt-get install -y cuda-toolkit-12-4 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY ./requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt


COPY ./datasets/CropOrWeed2 /app/datasets/CropOrWeed2
#COPY ./datasets/Coarse1 /app/datasets/Coarse1
COPY ./models/cow_default_best /app/models/cow_default_best
COPY ./models/yolo11n.pt /app/models/yo11n.pt
COPY ./scripts /app/scripts


CMD ["bash", "-c", "rm -rf /app/runs/detect/* && rm -rf /app/datasets/Coarse1 && tensorboard --logdir=/app/runs --host=0.0.0.0 --port=6006 & python3 /app/scripts/train_eval.py"]

#CMD ["bash", "-c", "tensorboard --logdir=/app/runs --host=0.0.0.0 --port=6006 & python3 /app/scripts/train_eval.py"]

