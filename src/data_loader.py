import pathlib
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import ImageFile

# Cho phép đọc các ảnh bị truncate trong dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True

from config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    IMG_SIZE, BATCH_SIZE, SEED,
)

# ── Transforms ───────────────────────────────────────────────────────────────
# ImageNet mean/std (dùng cho MobileNetV3 pre-trained)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
        ),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
        # RandomErasing
        # giảm overfitting 
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _make_loader(data_dir: pathlib.Path, transform, shuffle: bool) -> DataLoader:
    dataset = datasets.ImageFolder(root=str(data_dir), transform=transform)
    g = torch.Generator().manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,      
        pin_memory=True,
        generator=g if shuffle else None,
    )


def get_train_loader(train_dir=None) -> DataLoader:
    return _make_loader(
        train_dir or TRAIN_DIR,
        get_train_transform(),
        shuffle=True,
    )


def get_val_loader(val_dir=None) -> DataLoader:
    return _make_loader(
        val_dir or VAL_DIR,
        get_eval_transform(),
        shuffle=False,
    )


def get_test_loader(test_dir=None) -> DataLoader:
    return _make_loader(
        test_dir or TEST_DIR,
        get_eval_transform(),
        shuffle=False,
    )


def get_class_names(data_dir=None) -> list[str]:
    ds = datasets.ImageFolder(root=str(data_dir or TRAIN_DIR))
    return ds.classes
