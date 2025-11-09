import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
import requests
from io import BytesIO
import os

# ------------------------------
# Optional image loader for RGB images
# ------------------------------
def rgb_loader(path_or_url):
    """
    Load an image from local path or URL
    """
    if path_or_url.startswith("http"):
        response = requests.get(path_or_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        img = Image.open(path_or_url).convert("RGB")
    return img

# ------------------------------
# Supervised Contrastive Dataset Wrapper
# ------------------------------
class SupConDataset(Dataset):
    def __init__(self, base_dataset, transform=None):
        """
        Args:
            base_dataset: any PyTorch dataset returning (img, label)
            transform: augmentation transforms to apply
        """
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]

        # Apply transforms twice to generate two views
        if self.transform is not None:
            xi = self.transform(img)
            xj = self.transform(img)
        else:
            xi, xj = transforms.ToTensor()(img), transforms.ToTensor()(img)

        return (xi, xj), label, idx

# ------------------------------
# Utility to create datasets and dataloaders
# ------------------------------
def get_datasets_and_loaders(
    root=None,
    dataset_class=None,
    transform=None,
    batch_size=32,
    num_workers=4,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42
):
    """
    Create train/val/test datasets and dataloaders
    Args:
        root: local folder path for ImageFolder or dataset root
        dataset_class: a torch dataset class (e.g., datasets.CIFAR10)
        transform: augmentations / preprocessing
        batch_size: batch size for dataloaders
        num_workers: number of workers
        train_ratio, val_ratio: split ratios
        seed: random seed for splitting
    """
    # Load base dataset
    if dataset_class is not None:
        base_dataset = dataset_class(root=root, train=True, download=True, transform=None)
    elif root is not None:
        base_dataset = datasets.ImageFolder(root=root, transform=None, loader=rgb_loader)
    else:
        raise ValueError("Provide either root path or dataset_class")

    # Split into train/val/test
    total_len = len(base_dataset)
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    test_len = total_len - train_len - val_len

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        base_dataset, [train_len, val_len, test_len], generator=generator
    )

    # Wrap in SupConDataset to create two views
    train_dataset = SupConDataset(train_dataset, transform=transform)
    val_dataset = SupConDataset(val_dataset, transform=transform)
    test_dataset = SupConDataset(test_dataset, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
