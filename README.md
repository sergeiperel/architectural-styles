# Architectural Styles Classification
Architectural styles classification with neural networks.

The project implements a full ML pipeline:
data loading &#8594; preprocessing &#8594; training &#8594; inference &#8594; experiment logging
using PyTorch Lightning, Hydra, DVC, MLflow, and uv.

## 📕 Project Overview

The goal of the project is to classify images of architectural objects by architectural styles based on images of building facades.

Dataset:
- Building images are grouped by architectural style
- Data format:
    ```
    architectural-styles-dataset/
    ├── style_1/
    │ ├── img1.jpg
    │ ├── img2.jpg
    ├──style_2/
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
- model.name=lenet
- model.name=archinet


## ⚙️ Tech Stack

- Python 3.12
- PyTorch Lightning
- Hydra
- DVC
- MLflow
- uv (dependency management)
- pre-commit + ruff

## 🛠️ Setup

1. Cloning the repository
    ```
    git clone https://github.com/sergeiperel/architectural-styles.git
    cd architectural-styles
    ```

2. Creating a virtual environment and installing dependencies
    ```
    uv sync
    .\.venv\Scripts\activate.ps1
    ```

3. Installing pre-commit hooks
    ```
    uv run pre-commit install
    uv run pre-commit run -a
    ```

## 📦 Data Management (DVC)
Downloading a dataset. DVC will automatically download data from remote storage (Yandex S3).
```
uv run dvc pull
```

## 💪 Train
Don't forget to start MLflow server before the training part

```
mlflow ui --host 127.0.0.1 --port 8080
```

Training is launched through a single entry point using Hydra. Hyperparameters and model choice can be overridden directly in the command line or through configs.

- ArchiNet Training
    ```
    uv run python -m architectural_styles.train "model.name=archinet" "train.epochs=1" "train.lr=0.01" "data.batch_size=32"
    ```
- LeNet Training
    ```
    uv run python -m architectural_styles.train "model.name=lenet" "train.epochs=3" "train.lr=0.001" "data.batch_size=16"
    ```

What happens during training:
- Data loading
- Model training with validation
- Metrics logging via MLflow
- Checkpoints saved in checkpoints/{model_name}/ (each model has its own folder with last.ckpt and top-k checkpoints)


## 📊 Logging (MLflow)

During training, the following are logged:
- train_loss
- val_loss
- train_acc
- val_acc
- hyperparameters

Launching MLflow UI:
```
mlflow ui --host 127.0.0.1 --port 8080
```

After this, the interface will be accessible at: http://127.0.0.1:8080

## 🔍 Inference
Inference is performed using the last checkpoint of the trained model.
```
uv run python -m architectural_styles.infer model.name=archinet
```

The inference results are saved in a CSV file:
```
outputs/inference_results.csv
```

By default, the script automatically loads the latest (last.ckpt) checkpoint corresponding to the selected model.
If needed, you can explicitly specify a particular checkpoint to run inference with:
```
uv run python -m architectural_styles.infer model.name=lenet "infer.checkpoint_path=checkpoints/lenet/lenet_epoch-0_lr-0.01_val_loss-3.40.ckpt"
```

## 📂 Project Structure
```
architectural-styles/
├── checkpoints/            # saved checkpoints for each model
├── configs/                # Hydra configs
│   ├── data/
│   ├── infer/
│   ├── logging/
│   ├── model/
│   ├── train/
│   └── config.yaml
├── data/                   # DVC metadata + inference data
├── outputs/                # inference results & hydra logs
├── src/
│   └── architectural_styles/
│       ├── data/           # Dataset & dataloaders
│       ├── models/         # CNN models + LightningModule
│       ├── infer.py
│       └── train.py
├── uv.lock
├── data.dvc
├── pyproject.toml
├── LICENSE
└── README.md
```
