"""
Prototype Pipeline: BNN Training and Dirichlet Distribution Analysis

Trains a Bayesian Neural Network on MNIST and analyzes predictions using
Dirichlet distributions to assess plausibility of candidate distributions.
"""

import argparse

from model_checker.prototype_pipeline.pipeline import (
    primary_pipeline,
    variance_epoch_pipeline,
)


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
    parser.add_argument(
        "--salt-pepper-p",
        type=float,
        default=0.05,
        help="Probability for salt-and-pepper noise (default: 0.05)",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        nargs="+",
        default=["primary"],
        choices=["primary", "variance-epoch"],
        help="Pipelines to run: 'primary' and/or 'variance-epoch' (default: primary)",
    )
    parser.add_argument(
        "--variance-values",
        type=float,
        nargs="+",
        default=[8, 16],
        help="Variance values to test in variance-epoch pipeline",
    )
    parser.add_argument(
        "--epoch-values",
        type=int,
        nargs="+",
        default=[4, 6, 8],
        help="Epoch values to test in variance-epoch pipeline",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Target class to visualize in variance-epoch pipeline (default: same as digit)",
    )

    args = parser.parse_args()

    # Run primary pipeline if requested
    if "primary" in args.pipeline:
        primary_pipeline(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            model_path=args.model_path,
            device=args.device,
            kl_weight=args.kl_weight,
            train=args.train,
            digits=args.digits,
            lambda_concentration=args.lambda_concentration,
            k_std=args.k_std,
            samples=args.samples,
            output_dir=args.output_dir,
            seed=args.seed,
            salt_pepper_p=args.salt_pepper_p,
        )

    # Run variance-epoch pipeline if requested
    if "variance-epoch" in args.pipeline:
        variance_epoch_pipeline(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            kl_weight=args.kl_weight,
            digits=args.digits,
            lambda_concentration=args.lambda_concentration,
            k=args.k_std,
            samples=args.samples,
            output_dir=args.output_dir,
            seed=args.seed,
            variance_values=args.variance_values,
            epoch_values=args.epoch_values,
            target_class=args.target_class,
        )


if __name__ == "__main__":
    main()
