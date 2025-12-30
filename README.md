# Architectural Styles Classification
Architectural styles classification with neural networks.

The project implements a full ML pipeline:
data loading &#8594; preprocessing &#8594; training &#8594; inference &#8594; experiment logging
using PyTorch Lightning, Hydra, DVC, MLflow, and uv.

## Project Overview

The goal of the project is to classify images of architectural objects by architectural styles based on images of building facades.

Dataset

- Building images are grouped by architectural style
- Data format:
    ```
    architectural-styles-dataset/
    ├── style_1/
    │ ├── img1.jpg
    │ ├── img2.jpg
    ├── style_2/
    │ ├── img1.jpg
    │ ├── img2.jpg
    ...
    ```
- Data is not stored in Git.
- The dataset is managed via DVC.

## 🧠 Models
The project implements two neural networks (RGB, 256×256):

🔹 LeNet5RGB

Adaptation of the classic LeNet-5 architecture for RGB images.

🔹 ArchiNet (custom CNN)

A deeper custom convolutional network, developed specifically for this task.

Model selection is performed via Hydra:
```
- model.name=lenet
- model.name=archinet
```

## ⚙️ Tech Stack

- Python 3.12
- PyTorch Lightning
- Hydra
- DVC
- MLflow
- uv (dependency management)
- pre-commit + ruff

## 🛠 Setup

1. Cloning the repository
    ```
    git clone https://github.com/sergeiperel/architectural-styles.git
    cd architectural_styles
    ```

2. Creating a virtual environment and installing dependencies
    ```
    uv venv
    uv sync
    ```

3. Installing pre-commit hooks
    ```
    pre-commit install
    pre-commit run -a
    ```

## 📦 Data Management (DVC)
Downloading a dataset. DVC will automatically download data from remote storage (Yandex S3).
    ```
    dvc pull
    ```

## Train
Training is launched through a single entry point using Hydra

- ArchiNet Training
    ```
  python -m architectural_styles.main model.name=archinet
    ```
- LeNet Training
    ```
    python -m architectural_styles.main model.name=lenet
    ```

What happens during training:
- Data loading
- Model training
- Validation
- Metrics logging in MLflow
- Checkpoints saving in lightning_logs/


## 📊 Logging (MLflow)

During training, the following are logged:
- train_loss
- val_loss
- train_acc
- val_acc
- hyperparameters
- code version (git commit)

Launching MLflow UI:
    ```
    mlflow ui --host 127.0.0.1 --port 8080
    ```
After this, the interface will be accessible at:

👉 http://127.0.0.1:8080

## Inference
Inference is performed using the last checkpoint of the trained model.
    ```
    python -m architectural_styles.infer model.name=archinet
    ```

The inference results are saved in a CSV file:
    ```
    outputs/inference_results.csv
    ```

## 📂 Project Structure
```
architectural-styles/
├── configs/                # Hydra configs
├── data/                   # DVC metadata
├── src/
│   └── architectural_styles/
│       ├── data/           # Dataset & dataloaders
│       ├── models/         # CNN models + LightningModule
│       ├── preprocessing/  # Transforms
│       ├── infer.py
│       └── main.py
├── pyproject.toml
├── uv.lock
├── data.dvc
└── README.md
```
