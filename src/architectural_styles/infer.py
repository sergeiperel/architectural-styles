from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig

from architectural_styles.data.dataset import ArchitecturalStylesDataset
from architectural_styles.data.transform import get_val_transforms
from architectural_styles.models.module import ImageClassifier


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    model = ImageClassifier.load_from_checkpoint(
        cfg.infer.checkpoint_path,
        model_name=cfg.model.name,
        num_classes=cfg.model.num_classes,
        lr=cfg.train.lr,
    )
    model.eval()
    print(f"Loaded model: {cfg.model.name} from {cfg.infer.checkpoint_path}")

    results = []

    dataset = ArchitecturalStylesDataset(
        root_dir=cfg.data.infer_dir, transform=get_val_transforms()
    )

    for idx in range(len(dataset)):
        img, label = dataset[idx]
        img = img.unsqueeze(0)

        with torch.no_grad():
            logits = model(img)
            pred_class = logits.argmax(dim=1).item()

        results.append({"image_index": idx, "true_label": label, "pred_label": pred_class})

    output_path = Path(cfg.infer.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Inference finished! Results saved to {output_path}")


if __name__ == "__main__":
    main()
