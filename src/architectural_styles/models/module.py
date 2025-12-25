import lightning as L
import torch
import torch.nn.functional as F

from architectural_styles.models.archinet import ArchiNet
from architectural_styles.models.lenet import LeNet5RGB


class ImageClassifier(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        lr: float,
    ):
        super().__init__()
        self.save_hyperparameters()

        if model_name == "lenet":
            self.model = LeNet5RGB(num_classes)
        elif model_name == "archinet":
            self.model = ArchiNet(num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
