FROM python:3.9-slim



RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* \

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./datasets /app/datasets
COPY ./scripts /app/scripts
COPY ./models /app/models

# Standardkommando
CMD ["python", "scripts/train_eval.py"]
