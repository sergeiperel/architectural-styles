from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class ArchitecturalStylesDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"{self.root_dir} does not exist. Did you forget to run `dvc pull`?"
            )

        self.samples: list[tuple[Path, int]] = []
        self.class_to_idx: dict[str, int] = {}

        self._prepare_dataset()

    def _prepare_dataset(self) -> None:
        classes = sorted(d.name for d in self.root_dir.iterdir() if d.is_dir())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        for cls in classes:
            cls_dir = self.root_dir / cls
            for img_path in cls_dir.glob("*.jpg"):
                self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
