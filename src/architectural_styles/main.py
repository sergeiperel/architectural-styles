import hydra
import lightning as L
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

    trainer = L.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="auto",
        devices="auto",
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
