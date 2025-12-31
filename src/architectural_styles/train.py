from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig

from architectural_styles.data.dataloaders import create_dataloaders
from architectural_styles.models.module import ImageClassifier


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint_dir = Path(f"checkpoints/{cfg.model.name}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    last_ckpt = checkpoint_dir / "last.ckpt"
    if last_ckpt.exists():
        last_ckpt.unlink()

    train_loader, val_loader = create_dataloaders(
        data_dir=cfg.data.dir,
        batch_size=cfg.data.batch_size,
    )

    model = ImageClassifier(
        model_name=cfg.model.name,
        num_classes=cfg.model.num_classes,
        lr=cfg.train.lr,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=f"checkpoints/{cfg.model.name}",
        filename=(f"{cfg.model.name}_epoch-{{epoch}}_lr-{cfg.train.lr}_val_loss-{{val_loss:.2f}}"),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.logging.mlflow.experiment_name,
        tracking_uri=cfg.logging.mlflow.tracking_uri,
    )

    trainer = L.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="auto",
        devices="auto",
        logger=mlflow_logger,
        callbacks=[checkpoint_cb],
    )

    mlflow_logger.log_hyperparams(
        {
            "model_name": cfg.model.name,
            "lr": cfg.train.lr,
            "batch_size": cfg.data.batch_size,
            "epochs": cfg.train.epochs,
            "num_classes": cfg.model.num_classes,
        }
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
