import hydra
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig

from architectural_styles.data.dataloaders import create_dataloaders
from architectural_styles.models.module import ImageClassifier


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig):
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
        dirpath="checkpoints",
        filename=f"{cfg.model.name}-{{epoch}}-{{val_loss:.2f}}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    mlflow_logger = MLFlowLogger(
        experiment_name="architectural_styles",
        tracking_uri="http://127.0.0.1:8080",
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
