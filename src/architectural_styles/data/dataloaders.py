from torch.utils.data import DataLoader, random_split

from architectural_styles.data.dataset import ArchitecturalStylesDataset
from architectural_styles.data.transform import get_train_transforms, get_val_transforms


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 4,
):
    full_dataset = ArchitecturalStylesDataset(
        root_dir=data_dir,
        transform=get_train_transforms(),
    )

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    val_ds.dataset.transform = get_val_transforms()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader
