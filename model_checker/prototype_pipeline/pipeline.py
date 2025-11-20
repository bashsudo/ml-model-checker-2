from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from model_checker.model.bnn.prediction import make_prediction
from model_checker.prototype_pipeline.dirichlet import dirichlet_mean_variance
from model_checker.prototype_pipeline.visualization import visualize_dirichlet_pdf


def analyze_digit(
    model: nn.Module,
    test_dataset,
    digit: int,
    lambda_concentration: float = 10.0,
    noise_scale: float = 0.1,
    k: float = 3.0,
    samples: int = 20,
    device: str = "cpu",
    rng: Optional[np.random.Generator] = None,
    output_dir: str = "./experiment_results",
) -> Optional[np.ndarray]:
    """
    Analyze a single digit using Dirichlet distribution.

    Args:
        model: Trained BayesianNetwork model
        test_dataset: Test dataset
        digit: Digit to analyze (0-9)
        lambda_concentration: Concentration parameter for Dirichlet
        noise_scale: Scale of noise to add to prediction (unused, kept for compatibility)
        k: Number of standard deviations for threshold
        samples: Number of stochastic samples for prediction
        device: Device to run on
        rng: Random number generator
        output_dir: Directory to save plots

    Returns:
        Prediction distribution array
    """
    if rng is None:
        rng = np.random.default_rng()

    print(f"\n{'='*70}")
    print(f"ANALYZING DIGIT: {digit}")
    print(f"{'='*70}")

    # Group test data by label
    digit_indices = []
    for i in range(len(test_dataset)):
        record = test_dataset[i]
        if record["targets"] == digit:
            digit_indices.append(i)

    if not digit_indices:
        print(f"  ERROR: No test samples found for digit {digit}")
        return None

    # Get random sample
    sample_idx = rng.choice(digit_indices)
    record = test_dataset[sample_idx]
    image = record["inputs"]
    true_label = record["targets"]

    print(f"  Selected test sample index: {sample_idx}")
    print(f"  True label: {true_label}")

    # Ensure image is tensor and properly shaped
    if not torch.is_tensor(image):
        image = torch.tensor(image, dtype=torch.float32)
    if image.dim() == 1:
        image = image.view(1, -1)
    elif image.dim() == 2 and image.size(0) != 1:
        image = image.unsqueeze(0)

    # Save image
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    image_np = image.view(28, 28).cpu().numpy()
    plt.figure(figsize=(4, 4))
    plt.imshow(image_np, cmap="gray")
    plt.title(f"Test Image: Digit {true_label}")
    plt.axis("off")
    plt.savefig(f"{output_dir}/digit_{digit}_image.png", dpi=150, bbox_inches="tight")
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
    pdf_output_path = f"{output_dir}/digit_{digit}_dirichlet_pdf.png"
    visualize_dirichlet_pdf(
        alpha=alpha,
        digit=digit,
        k=k,
        output_path=pdf_output_path,
        rng=rng,
    )

    print(f"\n{'='*70}\n")

    return prediction
