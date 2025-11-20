"""
Prototype Pipeline: BNN Training and Dirichlet Distribution Analysis

Trains a Bayesian Neural Network on MNIST and analyzes predictions using
Dirichlet distributions to assess plausibility of candidate distributions.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model_checker.model.bnn import BayesianNetwork
from model_checker.model.bnn.dataset import create_dataloader, load_mnist
from model_checker.model.bnn.training import train_bnn
from model_checker.prototype_pipeline.pipeline import analyze_digit
from model_checker.umli.utils.device import get_device








def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(
        description="BNN Training and Dirichlet Distribution Analysis Pipeline"
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
        "--model-path",
        type=str,
        default="bayes_mnist.pth",
        help="Path to save/load model",
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
    parser.add_argument(
        "--train", action="store_true", help="Train the model (otherwise load existing)"
    )
    parser.add_argument(
        "--digits",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        help="Digits to analyze (default: all 0-9)",
    )
    parser.add_argument(
        "--lambda-concentration",
        type=float,
        default=10.0,
        help="Concentration parameter lambda for Dirichlet (default: 10.0)",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.1,
        help="Scale of noise to add to predictions (default: 0.1)",
    )
    parser.add_argument(
        "--k-std",
        type=float,
        default=3.0,
        help="Number of standard deviations for plausibility threshold (default: 3.0)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of stochastic samples for prediction (default: 20)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiment_results",
        help="Directory to save plots (default: ./experiment_results)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Get device
    device = args.device or get_device()
    print(f"{'='*70}")
    print(f"BNN TRAINING AND DIRICHLET ANALYSIS PIPELINE")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Random seed: {args.seed}")

    # Load datasets
    print(f"\n{'='*70}")
    print("LOADING DATASETS")
    print(f"{'='*70}")
    print("Loading MNIST training dataset...")
    train_dataset = load_mnist(root=args.data_dir, train=True, download=True)
    train_loader = create_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    print(f"  Training set size: {len(train_dataset)}")

    print("Loading MNIST test dataset...")
    test_dataset = load_mnist(root=args.data_dir, train=False, download=True)
    print(f"  Test set size: {len(test_dataset)}")

    # Create model
    print(f"\n{'='*70}")
    print("MODEL SETUP")
    print(f"{'='*70}")
    print("Creating Bayesian Neural Network...")
    model = BayesianNetwork()

    # Train or load model
    model_path = Path(args.model_path)
    if args.train or not model_path.exists():
        print(f"\n{'='*70}")
        print("TRAINING MODEL")
        print(f"{'='*70}")
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

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
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"\n  Model saved to: {model_path}")
        print(f"  Final loss: {losses[-1]:.4f}")
    else:
        print(f"\n{'='*70}")
        print("LOADING MODEL")
        print(f"{'='*70}")
        print(f"Loading model from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        print("  Model loaded successfully")

    # Analyze digits
    print(f"\n{'='*70}")
    print("DIRICHLET ANALYSIS")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Lambda (concentration): {args.lambda_concentration}")
    print(f"  Noise scale: {args.noise_scale}")
    print(f"  K (std dev threshold): {args.k_std}")
    print(f"  Prediction samples: {args.samples}")
    print(f"  Output directory: {args.output_dir}")

    for digit in args.digits:
        if digit not in range(10):
            print(f"\n  WARNING: Skipping invalid digit {digit} (must be 0-9)")
            continue

        analyze_digit(
            model=model,
            test_dataset=test_dataset,
            digit=digit,
            lambda_concentration=args.lambda_concentration,
            noise_scale=args.noise_scale,
            k=args.k_std,
            samples=args.samples,
            device=device,
            rng=rng,
            output_dir=args.output_dir,
        )

    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
