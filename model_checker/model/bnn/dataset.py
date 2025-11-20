"""
MNIST Dataset Loading

Loads MNIST dataset using torchvision and wraps it with UMLI's TorchDatasetAdapter.
"""

from typing import Any, List

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from model_checker.umli.data.adapters import TorchDatasetAdapter


class MNISTDataset(Dataset):
    """
    Wrapper for torchvision MNIST dataset that flattens images.
    """

    def __init__(self, mnist_dataset: datasets.MNIST):
        """
        Initialize MNIST dataset wrapper.

        Args:
            mnist_dataset: torchvision MNIST dataset instance
        """
        self.mnist_dataset = mnist_dataset

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.mnist_dataset)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get a data sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (flattened_image, label)
        """
        image, label = self.mnist_dataset[idx]
        # Flatten image: (1, 28, 28) -> (784,)
        image = image.view(-1)
        return image, label


def load_mnist(
    root: str = "./data",
    train: bool = True,
    download: bool = True,
    normalize: bool = True,
) -> TorchDatasetAdapter:
    """
    Load MNIST dataset and wrap with UMLI TorchDatasetAdapter.

    Args:
        root: Root directory for dataset storage
        train: If True, load training set; otherwise load test set
        download: If True, download dataset if not present
        normalize: If True, normalize images to [0, 1] range

    Returns:
        TorchDatasetAdapter wrapping the MNIST dataset
    """
    # Define transforms
    transform_list: List[Any] = [transforms.ToTensor()]
    if normalize:
        # ToTensor already normalizes to [0, 1], but we can add explicit normalization
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))

    transform = transforms.Compose(transform_list)

    # Load MNIST dataset
    mnist_dataset = datasets.MNIST(
        root=root, train=train, download=download, transform=transform
    )

    # Wrap with custom dataset that flattens images
    wrapped_dataset = MNISTDataset(mnist_dataset)

    # Wrap with UMLI TorchDatasetAdapter
    return TorchDatasetAdapter(wrapped_dataset)


def create_dataloader(
    dataset: TorchDatasetAdapter,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a PyTorch DataLoader from UMLI dataset.

    Args:
        dataset: UMLI TorchDatasetAdapter
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading

    Returns:
        PyTorch DataLoader
    """
    # Extract the underlying PyTorch dataset
    if hasattr(dataset, "dataset") and dataset.dataset is not None:
        torch_dataset = dataset.dataset
    else:
        # Convert to tensors if needed
        data, targets = dataset.to_torch()
        if targets is not None:
            torch_dataset = torch.utils.data.TensorDataset(data, targets)
        else:
            torch_dataset = torch.utils.data.TensorDataset(data)

    return DataLoader(
        torch_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
