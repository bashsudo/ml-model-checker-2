"""
Noise generation functions for image augmentation.

Provides functions for adding various types of noise to images,
specifically designed for MNIST dataset.
"""

from typing import Union

import numpy as np
import torch


def add_salt_pepper_noise(
    image: Union[torch.Tensor, np.ndarray],
    p: float = 0.05,
    rng: Union[np.random.Generator, None] = None,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Add salt and pepper noise to an image by inverting random pixels.

    For each pixel in the image, samples from a Bernoulli distribution with
    probability p. If the sample is 1 (heads), the pixel value is inverted.
    Otherwise, the pixel value remains unchanged.

    Args:
        image: Input image as torch.Tensor or np.ndarray.
               Can be flattened (784,) or 2D (28, 28) shape.
               Pixel values should be in [0, 1] range.
        p: Probability of inverting each pixel (default: 0.05)
        rng: Optional numpy random number generator for reproducibility.
             If None, uses default_rng().

    Returns:
        Noisy image in the same format and shape as input.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Convert to numpy if torch tensor
    is_torch = isinstance(image, torch.Tensor)
    if is_torch:
        image_np = image.cpu().numpy()
        original_shape = image.shape
    else:
        image_np = np.asarray(image)
        original_shape = image_np.shape

    # Sample from Bernoulli distribution for each pixel
    # Shape matches the image shape
    bernoulli_samples = rng.binomial(n=1, p=p, size=image_np.shape).astype(
        image_np.dtype
    )

    # Invert pixels where Bernoulli sample is 1
    # For pixel values in [0, 1] range: invert = 1 - pixel_value
    # Clamp to ensure values stay in [0, 1] range
    inverted_pixels = 1.0 - image_np
    noisy_image = np.where(bernoulli_samples == 1, inverted_pixels, image_np)
    noisy_image = np.clip(noisy_image, 0.0, 1.0)

    # Convert back to torch tensor if input was torch tensor
    if is_torch:
        return torch.from_numpy(noisy_image).to(image.dtype).to(image.device)
    else:
        return noisy_image
