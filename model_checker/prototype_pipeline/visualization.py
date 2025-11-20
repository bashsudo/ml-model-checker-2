from typing import Optional

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from model_checker.prototype_pipeline.dirichlet import dirichlet_mean_variance


def visualize_dirichlet_pdf(
    alpha: np.ndarray,
    digit: int,
    k: float,
    output_path: str,
    num_samples: int = 10000,
    rng: Optional[np.random.Generator] = None,
):
    """
    Visualize Dirichlet PDF.

    Creates a plot showing marginal distributions for each digit class,
    with k standard deviation thresholds marked.

    Args:
        alpha: Dirichlet concentration parameters
        digit: True digit label
        k: Number of standard deviations for threshold
        output_path: Path to save figure
        num_samples: Number of samples from Dirichlet for PDF estimation
        rng: Random number generator
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample from Dirichlet distribution
    dirichlet_samples = rng.dirichlet(alpha, size=num_samples)

    # Compute Dirichlet mean and standard deviation for each class
    mean, variance = dirichlet_mean_variance(alpha)
    std_dev = np.sqrt(variance)

    # Create figure with subplots for each digit class
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for class_idx in range(10):
        ax = axes[class_idx]

        # Plot histogram of marginal distribution for this class
        ax.hist(
            dirichlet_samples[:, class_idx],
            bins=50,
            density=True,
            alpha=0.6,
            color="blue",
            label="Dirichlet PDF",
        )

        # Add orange dashed lines for k standard deviations from mean
        mean_val = mean[class_idx]
        std_val = std_dev[class_idx]
        ax.axvline(
            mean_val + k * std_val,
            color="orange",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"±{k}σ" if class_idx == 0 else "",
        )
        ax.axvline(
            mean_val - k * std_val,
            color="orange",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
        )
        # Also mark the mean
        ax.axvline(
            mean_val,
            color="green",
            linestyle="-",
            linewidth=1.5,
            alpha=0.5,
            label="Mean" if class_idx == 0 else "",
        )

        ax.set_xlabel(f"P(Digit {class_idx})")
        ax.set_ylabel("Density")
        ax.set_title(f"Digit {class_idx}")
        ax.grid(True, alpha=0.3)
        if class_idx == 0:
            ax.legend()

    plt.suptitle(f"Dirichlet PDF for Digit {digit}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Dirichlet PDF plot saved to: {output_path}")


def visualize_dirichlet_pdf_single_digit(
    alpha: np.ndarray,
    digit: int,
    target_class: int,
    k: float,
    output_path: str,
    num_samples: int = 10000,
    rng: Optional[np.random.Generator] = None,
):
    """
    Visualize Dirichlet PDF for a single digit class.

    Creates a single plot showing the marginal distribution for one digit class,
    with k standard deviation thresholds marked.

    Args:
        alpha: Dirichlet concentration parameters
        digit: True digit label being analyzed
        target_class: Which digit class (0-9) to visualize
        k: Number of standard deviations for threshold
        output_path: Path to save figure
        num_samples: Number of samples from Dirichlet for PDF estimation
        rng: Random number generator
    """
    if rng is None:
        rng = np.random.default_rng()

    if target_class < 0 or target_class >= len(alpha):
        raise ValueError(f"target_class must be between 0 and {len(alpha)-1}")

    # Sample from Dirichlet distribution
    dirichlet_samples = rng.dirichlet(alpha, size=num_samples)

    # Compute Dirichlet mean and standard deviation for the target class
    mean, variance = dirichlet_mean_variance(alpha)
    std_dev = np.sqrt(variance)
    mean_val = mean[target_class]
    std_val = std_dev[target_class]

    # Create single figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram of marginal distribution for target class
    ax.hist(
        dirichlet_samples[:, target_class],
        bins=50,
        density=True,
        alpha=0.6,
        color="blue",
        label="Dirichlet PDF",
    )

    # Add orange dashed lines for k standard deviations from mean
    ax.axvline(
        mean_val + k * std_val,
        color="orange",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"±{k}σ threshold",
    )
    ax.axvline(
        mean_val - k * std_val,
        color="orange",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
    )
    # Also mark the mean
    ax.axvline(
        mean_val,
        color="green",
        linestyle="-",
        linewidth=1.5,
        alpha=0.5,
        label="Mean",
    )

    ax.set_xlabel(f"P(Digit {target_class})")
    ax.set_ylabel("Density")
    ax.set_title(f"Dirichlet PDF for Digit {target_class} (Analyzing Digit {digit})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Dirichlet PDF plot (single digit) saved to: {output_path}")


def visualize_dirichlet_pdf_single_digit_by_variance(
    alphas_by_variance: dict[float, np.ndarray],
    digit: int,
    target_class: int,
    k: float,
    output_path: str,
    num_samples: int = 10000,
    rng: Optional[np.random.Generator] = None,
):
    """
    Visualize Dirichlet PDF for a single digit class across different variance values.

    For a particular epoch, shows how the PDFs vary by variance.
    Each PDF is plotted with transparency and a color corresponding to its variance value.

    Args:
        alphas_by_variance: Dictionary mapping variance values to Dirichlet alpha parameters
        digit: True digit label being analyzed
        target_class: Which digit class (0-9) to visualize
        k: Number of standard deviations for threshold
        output_path: Path to save figure
        num_samples: Number of samples from Dirichlet for PDF estimation
        rng: Random number generator
    """
    if rng is None:
        rng = np.random.default_rng()

    if target_class < 0 or target_class >= len(list(alphas_by_variance.values())[0]):
        raise ValueError(
            f"target_class must be between 0 and {len(list(alphas_by_variance.values())[0])-1}"
        )

    # Get variance values and sort them
    variances = sorted(alphas_by_variance.keys())

    # Create colormap for variances
    cmap = cm.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=min(variances), vmax=max(variances))
    colors = [cmap(norm(v)) for v in variances]

    # Create single figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot PDF for each variance
    for i, variance in enumerate(variances):
        alpha = alphas_by_variance[variance]

        # Sample from Dirichlet distribution
        dirichlet_samples = rng.dirichlet(alpha, size=num_samples)

        # Plot histogram with transparency and color
        ax.hist(
            dirichlet_samples[:, target_class],
            bins=50,
            density=True,
            alpha=0.4,
            color=colors[i],
            label=f"Variance={variance}",
        )

    # Compute mean and std for the first variance (for reference lines)
    first_alpha = alphas_by_variance[variances[0]]
    mean, variance_stats = dirichlet_mean_variance(first_alpha)
    std_dev = np.sqrt(variance_stats)
    mean_val = mean[target_class]
    std_val = std_dev[target_class]

    # Add orange dashed lines for k standard deviations from mean
    ax.axvline(
        mean_val + k * std_val,
        color="orange",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"±{k}σ threshold",
    )
    ax.axvline(
        mean_val - k * std_val,
        color="orange",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
    )
    # Also mark the mean
    ax.axvline(
        mean_val,
        color="green",
        linestyle="-",
        linewidth=1.5,
        alpha=0.5,
        label="Mean (reference)",
    )

    ax.set_xlabel(f"P(Digit {target_class})")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Dirichlet PDF for Digit {target_class} by Variance (Analyzing Digit {digit})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Dirichlet PDF plot (by variance) saved to: {output_path}")


def visualize_dirichlet_pdf_single_digit_by_epoch(
    alphas_by_epoch: dict[int, np.ndarray],
    digit: int,
    target_class: int,
    k: float,
    output_path: str,
    num_samples: int = 10000,
    rng: Optional[np.random.Generator] = None,
):
    """
    Visualize Dirichlet PDF for a single digit class across different epoch values.

    For a particular variance, shows how the PDFs vary by epoch.
    Each PDF is plotted with transparency and a color corresponding to its epoch value.

    Args:
        alphas_by_epoch: Dictionary mapping epoch values to Dirichlet alpha parameters
        digit: True digit label being analyzed
        target_class: Which digit class (0-9) to visualize
        k: Number of standard deviations for threshold
        output_path: Path to save figure
        num_samples: Number of samples from Dirichlet for PDF estimation
        rng: Random number generator
    """
    if rng is None:
        rng = np.random.default_rng()

    if target_class < 0 or target_class >= len(list(alphas_by_epoch.values())[0]):
        raise ValueError(
            f"target_class must be between 0 and {len(list(alphas_by_epoch.values())[0])-1}"
        )

    # Get epoch values and sort them
    epochs = sorted(alphas_by_epoch.keys())

    # Create colormap for epochs
    cmap = cm.get_cmap("plasma")
    norm = mcolors.Normalize(vmin=min(epochs), vmax=max(epochs))
    colors = [cmap(norm(e)) for e in epochs]

    # Create single figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot PDF for each epoch
    for i, epoch in enumerate(epochs):
        alpha = alphas_by_epoch[epoch]

        # Sample from Dirichlet distribution
        dirichlet_samples = rng.dirichlet(alpha, size=num_samples)

        # Plot histogram with transparency and color
        ax.hist(
            dirichlet_samples[:, target_class],
            bins=50,
            density=True,
            alpha=0.4,
            color=colors[i],
            label=f"Epoch={epoch}",
        )

    # Compute mean and std for the first epoch (for reference lines)
    first_alpha = alphas_by_epoch[epochs[0]]
    mean, variance_stats = dirichlet_mean_variance(first_alpha)
    std_dev = np.sqrt(variance_stats)
    mean_val = mean[target_class]
    std_val = std_dev[target_class]

    # Add orange dashed lines for k standard deviations from mean
    ax.axvline(
        mean_val + k * std_val,
        color="orange",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"±{k}σ threshold",
    )
    ax.axvline(
        mean_val - k * std_val,
        color="orange",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
    )
    # Also mark the mean
    ax.axvline(
        mean_val,
        color="green",
        linestyle="-",
        linewidth=1.5,
        alpha=0.5,
        label="Mean (reference)",
    )

    ax.set_xlabel(f"P(Digit {target_class})")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Dirichlet PDF for Digit {target_class} by Epoch (Analyzing Digit {digit})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Dirichlet PDF plot (by epoch) saved to: {output_path}")
