from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model_checker.model.bnn import BayesianNetwork
from model_checker.model.bnn.dataset import create_dataloader, load_mnist
from model_checker.model.bnn.prediction import make_prediction
from model_checker.model.bnn.training import train_bnn
from model_checker.prototype_pipeline.dirichlet import dirichlet_mean_variance
from model_checker.prototype_pipeline.noise import add_salt_pepper_noise
from model_checker.prototype_pipeline.visualization import (
    visualize_dirichlet_pdf,
    visualize_dirichlet_pdf_single_digit_by_epoch,
    visualize_dirichlet_pdf_single_digit_by_variance,
)
from model_checker.umli.utils.device import get_device


def sample_image_by_digit(
    test_dataset, digit: int, rng: Optional[np.random.Generator] = None
) -> Tuple[torch.Tensor, int]:
    """
    Sample a random image from the test dataset for a given digit.

    Args:
        test_dataset: Test dataset
        digit: Digit to sample (0-9)
        rng: Optional random number generator

    Returns:
        Tuple of (image_tensor, true_label)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Group test data by label
    digit_indices = []
    for i in range(len(test_dataset)):
        record = test_dataset[i]
        if record["targets"] == digit:
            digit_indices.append(i)

    if not digit_indices:
        raise ValueError(f"No test samples found for digit {digit}")

    # Get random sample
    sample_idx = rng.choice(digit_indices)
    record = test_dataset[sample_idx]
    image = record["inputs"]
    true_label = record["targets"]

    # Ensure image is tensor
    if not torch.is_tensor(image):
        image = torch.tensor(image, dtype=torch.float32)

    return image, true_label


def analyze_digit(
    model: nn.Module,
    image: torch.Tensor,
    true_label: int,
    digit: int,
    lambda_concentration: float = 10.0,
    k: float = 3.0,
    samples: int = 20,
    device: str = "cpu",
    rng: Optional[np.random.Generator] = None,
    output_dir: str = "./experiment_results",
    prefix: str = "",
) -> Optional[np.ndarray]:
    """
    Analyze a single digit image using Dirichlet distribution.

    Args:
        model: Trained BayesianNetwork model
        image: Input image tensor
        true_label: True label of the image
        digit: Digit being analyzed (0-9)
        lambda_concentration: Concentration parameter for Dirichlet
        k: Number of standard deviations for threshold
        samples: Number of stochastic samples for prediction
        device: Device to run on
        rng: Random number generator
        output_dir: Directory to save plots
        prefix: Prefix to add to output filenames (e.g., "pas_" for salt-and-pepper)

    Returns:
        Prediction distribution array
    """
    if rng is None:
        rng = np.random.default_rng()

    print(f"\n{'='*70}")
    print(f"ANALYZING DIGIT: {digit}" + (f" ({prefix})" if prefix else ""))
    print(f"{'='*70}")
    print(f"  True label: {true_label}")

    # Ensure image is properly shaped for model input
    if image.dim() == 1:
        image = image.view(1, -1)
    elif image.dim() == 2 and image.size(0) != 1:
        image = image.unsqueeze(0)

    # Save image
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    image_np = image.view(28, 28).cpu().numpy()
    prefix_str = f"{prefix}_" if prefix else ""
    plt.figure(figsize=(4, 4))
    plt.imshow(image_np, cmap="gray")
    plt.title(f"Test Image: Digit {true_label}" + (f" ({prefix})" if prefix else ""))
    plt.axis("off")
    plt.savefig(
        f"{output_dir}/{prefix_str}digit_{digit}_image.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # Get model prediction
    print(f"\n  Getting model prediction (using {samples} stochastic samples)...")
    model.eval()
    mean_probs, _ = make_prediction(model, image, samples=samples, device=device)
    prediction = mean_probs[0].cpu().numpy()  # Shape: (10,)

    pred_digit = np.argmax(prediction)
    pred_confidence = prediction[pred_digit]

    print(f"  Predicted digit: {pred_digit} (confidence: {pred_confidence:.4f})")
    print(f"  Prediction distribution: {prediction}")

    # Create Dirichlet distribution with prediction as mean
    # alpha = lambda * prediction (where lambda is concentration parameter)
    alpha = lambda_concentration * prediction
    print(f"\n  Dirichlet parameters:")
    print(f"    Lambda (concentration): {lambda_concentration}")
    print(f"    Alpha: {alpha}")

    # Compute Dirichlet statistics
    mean, variance = dirichlet_mean_variance(alpha)
    std_dev = np.sqrt(variance)
    print(f"    Mean: {mean}")
    print(f"    Std Dev: {std_dev}")

    # Visualize Dirichlet PDF
    print(f"\n  Generating Dirichlet PDF visualization...")
    prefix_str = f"{prefix}_" if prefix else ""
    pdf_output_path = f"{output_dir}/{prefix_str}digit_{digit}_dirichlet_pdf.png"
    visualize_dirichlet_pdf(
        alpha=alpha,
        digit=digit,
        k=k,
        output_path=pdf_output_path,
        rng=rng,
    )

    print(f"\n{'='*70}\n")

    return prediction


def primary_pipeline(
    data_dir: str = "./data",
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 0.001,
    model_path: str = "bayes_mnist.pth",
    device: Optional[str] = None,
    kl_weight: Optional[float] = None,
    train: bool = False,
    digits: Optional[List[int]] = None,
    lambda_concentration: float = 10.0,
    k_std: float = 3.0,
    samples: int = 20,
    output_dir: str = "./experiment_results",
    seed: int = 42,
    salt_pepper_p: float = 0.05,
    models_dir: str = "models",
    variance: float = 1.0,
):
    """
    Primary pipeline for BNN training and Dirichlet distribution analysis.

    Trains or loads a Bayesian Neural Network, then analyzes digits using
    Dirichlet distributions. For each digit, analyzes both the original image
    and a salt-and-pepper noisy version.

    Args:
        data_dir: Directory for MNIST data
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        model_path: Path to save/load model (deprecated, use models_dir instead)
        device: Device to use (cpu/cuda), None for auto-detect
        kl_weight: KL divergence weight (default: 1/N)
        train: If True, train the model (otherwise load existing)
        digits: List of digits to analyze (default: all 0-9)
        lambda_concentration: Concentration parameter for Dirichlet
        k_std: Number of standard deviations for threshold
        samples: Number of stochastic samples for prediction
        output_dir: Directory to save plots
        seed: Random seed for reproducibility
        salt_pepper_p: Probability for salt-and-pepper noise
        models_dir: Base directory for saving models (default: "models")
        variance: Variance value for the model (default: 1.0)
    """
    if digits is None:
        digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Get device
    device = device or get_device()
    print(f"{'='*70}")
    print(f"BNN TRAINING AND DIRICHLET ANALYSIS PIPELINE")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Random seed: {seed}")

    # Load datasets
    print(f"\n{'='*70}")
    print("LOADING DATASETS")
    print(f"{'='*70}")
    print("Loading MNIST training dataset...")
    train_dataset = load_mnist(root=data_dir, train=True, download=True)
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"  Training set size: {len(train_dataset)}")

    print("Loading MNIST test dataset...")
    test_dataset = load_mnist(root=data_dir, train=False, download=True)
    print(f"  Test set size: {len(test_dataset)}")

    # Create model
    print(f"\n{'='*70}")
    print("MODEL SETUP")
    print(f"{'='*70}")
    print("Creating Bayesian Neural Network...")
    model = BayesianNetwork(variance=variance)

    # Determine model save path (structured path)
    models_base_dir = Path(models_dir)
    models_subdir = models_base_dir / "primary"
    model_filename = f"bnn_seed_{seed}_var_{variance}_epoch_{epochs}.pth"
    model_path_obj = models_subdir / model_filename

    # Train or load model
    if train or not model_path_obj.exists():
        print(f"\n{'='*70}")
        print("TRAINING MODEL")
        print(f"{'='*70}")
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        losses = train_bnn(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=epochs,
            device=device,
            kl_weight=kl_weight,
            verbose=True,
        )

        # Save model to structured path
        model_path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path_obj)
        print(f"\n  Model saved to: {model_path_obj}")
        print(f"  Final loss: {losses[-1]:.4f}")
    else:
        print(f"\n{'='*70}")
        print("LOADING MODEL")
        print(f"{'='*70}")
        print(f"Loading model from: {model_path_obj}")
        model.load_state_dict(torch.load(model_path_obj, map_location=device))
        model.to(device)
        print("  Model loaded successfully")

    # Analyze digits
    print(f"\n{'='*70}")
    print("DIRICHLET ANALYSIS")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Lambda (concentration): {lambda_concentration}")
    print(f"  K (std dev threshold): {k_std}")
    print(f"  Prediction samples: {samples}")
    print(f"  Output directory: {output_dir}")
    print(f"  Salt-and-pepper noise probability: {salt_pepper_p}")

    for digit in digits:
        if digit not in range(10):
            print(f"\n  WARNING: Skipping invalid digit {digit} (must be 0-9)")
            continue

        # Sample image for this digit
        try:
            image, true_label = sample_image_by_digit(test_dataset, digit, rng=rng)
        except ValueError as e:
            print(f"\n  ERROR: {e}")
            continue

        # Analyze original image
        analyze_digit(
            model=model,
            image=image,
            true_label=true_label,
            digit=digit,
            lambda_concentration=lambda_concentration,
            k=k_std,
            samples=samples,
            device=device,
            rng=rng,
            output_dir=output_dir,
            prefix="",
        )

        # Create salt-and-pepper noisy version
        noisy_image = add_salt_pepper_noise(image, p=salt_pepper_p, rng=rng)
        # Ensure it's a torch tensor
        if not torch.is_tensor(noisy_image):
            noisy_image = torch.tensor(noisy_image, dtype=torch.float32)

        # Analyze noisy image
        analyze_digit(
            model=model,
            image=noisy_image,
            true_label=true_label,
            digit=digit,
            lambda_concentration=lambda_concentration,
            k=k_std,
            samples=samples,
            device=device,
            rng=rng,
            output_dir=output_dir,
            prefix="pas",
        )

    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}\n")


def variance_epoch_pipeline(
    data_dir: str = "./data",
    batch_size: int = 128,
    lr: float = 0.001,
    device: Optional[str] = None,
    kl_weight: Optional[float] = None,
    digits: Optional[List[int]] = None,
    lambda_concentration: float = 10.0,
    k: float = 3.0,
    samples: int = 20,
    output_dir: str = "./experiment_results",
    seed: int = 42,
    variance_values: Optional[List[float]] = None,
    epoch_values: Optional[List[int]] = None,
    target_class: Optional[int] = None,
    visualization_types: Optional[set[str]] = None,
    models_dir: str = "models",
):
    """
    Variance-epoch pipeline for analyzing how variance and epochs affect Dirichlet PDFs.

    Trains multiple models with different variance values and epoch counts,
    then visualizes how the Dirichlet PDFs vary across these parameters.

    Args:
        data_dir: Directory for MNIST data
        batch_size: Batch size for training
        lr: Learning rate
        device: Device to use (cpu/cuda), None for auto-detect
        kl_weight: KL divergence weight (default: 1/N)
        digits: List of digits to analyze (default: [0])
        lambda_concentration: Concentration parameter for Dirichlet
        k: Number of standard deviations for threshold
        samples: Number of stochastic samples for prediction
        output_dir: Directory to save plots
        seed: Random seed for reproducibility
        variance_values: List of variance values to test (default: [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024])
        epoch_values: List of epoch values to test (default: [2, 4, 6, 8])
        target_class: Which digit class to visualize (default: same as digit being analyzed)
        visualization_types: Set of visualization types to generate: {"variance", "epoch"} or both
                             (default: {"variance", "epoch"} - generates both)
        models_dir: Base directory for saving models (default: "models")
    """
    if variance_values is None:
        variance_values = [1, 16]
    if epoch_values is None:
        epoch_values = [4, 6, 8]
    if digits is None:
        digits = [0]
    if visualization_types is None:
        visualization_types = {"variance", "epoch"}

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Get device
    device = device or get_device()
    print(f"{'='*70}")
    print(f"VARIANCE-EPOCH PIPELINE")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Random seed: {seed}")
    print(f"Variance values: {variance_values}")
    print(f"Epoch values: {epoch_values}")

    # Load datasets
    print(f"\n{'='*70}")
    print("LOADING DATASETS")
    print(f"{'='*70}")
    print("Loading MNIST training dataset...")
    train_dataset = load_mnist(root=data_dir, train=True, download=True)
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"  Training set size: {len(train_dataset)}")

    print("Loading MNIST test dataset...")
    test_dataset = load_mnist(root=data_dir, train=False, download=True)
    print(f"  Test set size: {len(test_dataset)}")

    # Set up models directory
    models_base_dir = Path(models_dir)
    models_subdir = models_base_dir / "var-and-epoch"
    models_subdir.mkdir(parents=True, exist_ok=True)

    # Analyze each digit
    for digit in digits:
        if digit not in range(10):
            print(f"\n  WARNING: Skipping invalid digit {digit} (must be 0-9)")
            continue

        if target_class is None:
            target_class_to_use = digit
        else:
            target_class_to_use = target_class

        print(f"\n{'='*70}")
        print(f"ANALYZING DIGIT: {digit}")
        print(f"{'='*70}")

        # Sample image for this digit
        try:
            image, true_label = sample_image_by_digit(test_dataset, digit, rng=rng)
        except ValueError as e:
            print(f"\n  ERROR: {e}")
            continue

        # Ensure image is properly shaped
        if image.dim() == 1:
            image = image.view(1, -1)
        elif image.dim() == 2 and image.size(0) != 1:
            image = image.unsqueeze(0)

        # Store alphas for visualization
        alphas_by_variance_for_epoch: dict[int, dict[float, np.ndarray]] = {}
        alphas_by_epoch_for_variance: dict[float, dict[int, np.ndarray]] = {}

        # For each variance value
        for variance in variance_values:
            print(f"\n  {'-'*68}")
            print(f"  Variance: {variance}")
            print(f"  {'-'*68}")

            # Create model with this variance
            model = BayesianNetwork(variance=variance)
            model.to(device)

            # Store alphas for this variance across epochs
            alphas_by_epoch_for_variance[variance] = {}

            # Track previous epoch to enable incremental training
            previous_epochs = 0

            # For each epoch value
            for epochs in epoch_values:
                # Determine model save path
                model_filename = f"bnn_seed_{seed}_var_{variance}_epoch_{epochs}.pth"
                model_path_obj = models_subdir / model_filename

                # Try to load existing model
                if model_path_obj.exists():
                    print(f"\n    Loading existing model for {epochs} epochs...")
                    model.load_state_dict(
                        torch.load(model_path_obj, map_location=device)
                    )
                    model.to(device)
                    print(f"      Model loaded from: {model_path_obj}")
                    previous_epochs = epochs
                else:
                    # Check if we can continue from a previous epoch model
                    epochs_to_train = epochs - previous_epochs
                    
                    if epochs_to_train > 0:
                        print(f"\n    Training for {epochs_to_train} more epochs (total: {epochs})...")

                        # Create fresh optimizer for each training run
                        optimizer = optim.Adam(model.parameters(), lr=lr)
                        criterion = nn.CrossEntropyLoss()

                        # Train model incrementally
                        train_bnn(
                            model=model,
                            train_loader=train_loader,
                            optimizer=optimizer,
                            criterion=criterion,
                            epochs=epochs_to_train,
                            device=device,
                            kl_weight=kl_weight,
                            verbose=False,
                        )

                        # Save model
                        torch.save(model.state_dict(), model_path_obj)
                        print(f"      Model saved to: {model_path_obj}")
                        previous_epochs = epochs
                    else:
                        print(f"\n    Model already trained for {epochs} epochs, skipping...")

                # Get prediction
                model.eval()
                mean_probs, _ = make_prediction(
                    model, image, samples=samples, device=device
                )
                prediction = mean_probs[0].cpu().numpy()

                # Create Dirichlet alpha
                alpha = lambda_concentration * prediction

                # Store alpha
                alphas_by_epoch_for_variance[variance][epochs] = alpha

                # Store for variance-by-epoch visualization
                if epochs not in alphas_by_variance_for_epoch:
                    alphas_by_variance_for_epoch[epochs] = {}
                alphas_by_variance_for_epoch[epochs][variance] = alpha

                print(f"      Completed: variance={variance}, epochs={epochs}")

        # Create visualizations
        print(f"\n  {'-'*68}")
        print(f"  GENERATING VISUALIZATIONS")
        print(f"  {'-'*68}")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # For each epoch, visualize by variance (if requested)
        if "variance" in visualization_types:
            for epoch in epoch_values:
                if epoch in alphas_by_variance_for_epoch:
                    output_path = f"{output_dir}/digit_{digit}_epoch_{epoch}_by_variance_class_{target_class_to_use}.png"
                    visualize_dirichlet_pdf_single_digit_by_variance(
                        alphas_by_variance=alphas_by_variance_for_epoch[epoch],
                        digit=digit,
                        target_class=target_class_to_use,
                        k=k,
                        output_path=output_path,
                        rng=rng,
                    )

        # For each variance, visualize by epoch (if requested)
        if "epoch" in visualization_types:
            for variance in variance_values:
                if variance in alphas_by_epoch_for_variance:
                    output_path = f"{output_dir}/digit_{digit}_variance_{variance}_by_epoch_class_{target_class_to_use}.png"
                    visualize_dirichlet_pdf_single_digit_by_epoch(
                        alphas_by_epoch=alphas_by_epoch_for_variance[variance],
                        digit=digit,
                        target_class=target_class_to_use,
                        k=k,
                        output_path=output_path,
                        rng=rng,
                    )

    print(f"\n{'='*70}")
    print("VARIANCE-EPOCH PIPELINE COMPLETE")
    print(f"{'='*70}\n")
