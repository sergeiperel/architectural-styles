from torchvision import transforms


def get_train_transforms():
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )


def get_val_transforms():
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
