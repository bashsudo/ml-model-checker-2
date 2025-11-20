"""
Main Training Script for Bayesian Neural Network

Trains a BNN on MNIST using UMLI infrastructure.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from model_checker.model.bnn import BayesianNetwork
from model_checker.model.bnn.dataset import create_dataloader, load_mnist
from model_checker.model.bnn.training import train_bnn
from model_checker.umli.utils.device import get_device


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train Bayesian Neural Network on MNIST"
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data", help="Directory for MNIST data"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--output", type=str, default="bayes_mnist.pth", help="Output model path"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cpu/cuda)"
    )
    parser.add_argument(
        "--kl-weight",
        type=float,
        default=None,
        help="KL divergence weight (default: 1/N)",
    )

    args = parser.parse_args()

    # Get device
    device = args.device or get_device()
    print(f"Using device: {device}")

    # Load MNIST dataset using UMLI
    print("Loading MNIST training dataset...")
    train_dataset = load_mnist(root=args.data_dir, train=True, download=True)
    train_loader = create_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    print(f"Training set size: {len(train_dataset)}")

    # Create model
    print("Creating Bayesian Neural Network...")
    model = BayesianNetwork()

    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Train model
    print("Starting training...")
    losses = train_bnn(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=args.epochs,
        device=device,
        kl_weight=args.kl_weight,
        verbose=True,
    )

    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"\nTraining complete. Model saved to {output_path}")

    print(f"\nFinal loss: {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
