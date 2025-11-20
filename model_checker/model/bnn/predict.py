"""
Interactive Prediction Script for Bayesian Neural Network

Loads trained model and allows interactive testing with uncertainty visualization.
"""

import argparse
import random

import matplotlib.pyplot as plt
import torch

from model_checker.model.bnn import BayesianNetwork
from model_checker.model.bnn.dataset import load_mnist
from model_checker.model.bnn.prediction import get_top_predictions, predict_digit
from model_checker.umli.utils.device import get_device


def group_by_label(dataset):
    """
    Group dataset samples by label.

    Args:
        dataset: UMLI TorchDatasetAdapter

    Returns:
        Dictionary mapping label to list of sample indices
    """
    grouped = {i: [] for i in range(10)}

    for i in range(len(dataset)):
        record = dataset[i]
        label = (
            record["targets"].item()
            if torch.is_tensor(record["targets"])
            else record["targets"]
        )
        grouped[label].append(i)

    return grouped


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(
        description="Interactive Bayesian Neural Network prediction"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model (.pth file)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data", help="Directory for MNIST data"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of stochastic samples for prediction",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cpu/cuda)"
    )

    args = parser.parse_args()

    # Get device
    device = args.device or get_device()
    print(f"Using device: {device}")

    # Load test dataset
    print("Loading MNIST test dataset...")
    test_dataset = load_mnist(root=args.data_dir, train=False, download=True)
    print(f"Test set size: {len(test_dataset)}")

    # Group test data by label for easy sampling
    print("Grouping test data by label...")
    test_groups = group_by_label(test_dataset)

    # Load model
    print(f"Loading model from {args.model}...")
    model = BayesianNetwork()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    print("\n" + "=" * 50)
    print("Interactive Prediction Mode")
    print("Enter a digit (0-9) to test, or 'q' to quit")
    print("=" * 50)

    # Interactive testing loop
    while True:
        user_input = input("\nTest Digit: ").strip()

        if user_input.lower() == "q" or user_input == "":
            break

        if not user_input.isdigit() or int(user_input) not in range(10):
            print("Please enter a valid digit 0-9 ('q' to quit)")
            continue

        digit = int(user_input)

        # Sample a random image of the requested digit
        if not test_groups[digit]:
            print(f"No samples available for digit {digit}")
            continue

        sample_idx = random.choice(test_groups[digit])
        record = test_dataset[sample_idx]

        # Get image and label
        image = record["inputs"]
        true_label = record["targets"]

        # Ensure image is a tensor and reshape if needed
        if not torch.is_tensor(image):
            image = torch.tensor(image, dtype=torch.float32)
        if image.dim() == 1:
            image = image.view(1, -1)
        elif image.dim() == 2 and image.size(0) != 1:
            image = image.unsqueeze(0)

        # Make prediction
        pred, confidence, uncertainty = predict_digit(
            model, image, samples=args.samples, device=device
        )

        # Get top-2 predictions
        top_predictions = get_top_predictions(
            model, image, k=2, samples=args.samples, device=device
        )

        print(f"\nTrue Label: {true_label}")
        print(f"Predicted: {pred} with {confidence * 100:.2f}% confidence")
        print(f"Uncertainty (std): {uncertainty:.4f}")

        if len(top_predictions) > 1:
            pred2, conf2 = top_predictions[1]
            print(f"Second most likely: {pred2} with {conf2 * 100:.2f}% confidence")

        # Visualize prediction
        image_np = image.view(28, 28).cpu().numpy()
        plt.figure(figsize=(6, 6))
        plt.imshow(image_np, cmap="gray")
        plt.title(
            f"True: {true_label} | Predicted: {pred} | Confidence: {confidence * 100:.2f}%"
        )
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
