from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


GTSRB_MEAN = (0.3337, 0.3064, 0.3171)
GTSRB_STD = (0.2672, 0.2564, 0.2629)


def get_transforms(image_size=48):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(12),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.08, 0.08),
                scale=(0.90, 1.10),
                shear=5,
            ),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(GTSRB_MEAN, GTSRB_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(GTSRB_MEAN, GTSRB_STD),
        ]
    )
    return train_transform, eval_transform


def create_loaders(
    data_dir="data",
    image_size=48,
    batch_size=128,
    num_workers=2,
    seed=42,
    validation_ratio=0.1,
):
    data_dir = Path(data_dir)
    train_transform, eval_transform = get_transforms(image_size)

    full_train_aug = datasets.GTSRB(
        root=data_dir, split="train", download=True, transform=train_transform
    )
    full_train_eval = datasets.GTSRB(
        root=data_dir, split="train", download=True, transform=eval_transform
    )
    test_set = datasets.GTSRB(
        root=data_dir, split="test", download=True, transform=eval_transform
    )

    val_size = int(len(full_train_aug) * validation_ratio)
    train_size = len(full_train_aug) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset_aug = random_split(
        full_train_aug, [train_size, val_size], generator=generator
    )
    val_subset = torch.utils.data.Subset(full_train_eval, val_subset_aug.indices)

    loader_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_subset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)
    return train_loader, val_loader, test_loader
