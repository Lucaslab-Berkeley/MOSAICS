from typing import Literal

import numpy as np
import torch


def cross_correlate_same_shape(
    image: np.ndarray, template: np.ndarray
) -> float:
    """Cross-correlated an image and template (both provided in Fourier space)
    which have the same shape. The result will be a single scalar value.

    Args:
        image (np.ndarray): The FFT of the image.
        template (np.ndarray): The FFT of the template.
    """
    return np.sum(image * np.conj(template))


def cross_correlate_from_fft(
    image_fft: np.ndarray,
    template_fft: np.ndarray,
    mode: Literal["valid", "full"] = "valid",
) -> np.ndarray:
    """Cross-correlates an image and template (both provided in Fourier space)
    with the provided mode. If mode is 'valid', then the result is cropped to
    be the valid bounds of cross-correlation. If mode is 'full', then the
    result is not cropped.

    Args:
        image_fft (np.ndarray): The FFT and shifted image.
        template_fft (np.ndarray): The FFT and shifted template.
        mode (Literal["valid", "full"]): The mode of the cross-correlation.
        Default is 'valid'.

    Returns:
        np.ndarray: The cross-correlation result.
    """
    out_shape = np.array(image_fft.shape) - np.array(template_fft.shape) + 1

    # FFT pad up to the same shape as the reference image
    template = np.fft.ifftshift(template_fft)
    template = np.fft.ifftn(template)
    template = np.fft.ifftshift(template)  # slicing is shifted in real space
    template_fft = np.fft.fftn(template, s=image_fft.shape)
    template_fft = np.fft.fftshift(template_fft)

    cross_correlation = image_fft * np.conj(template_fft)
    cross_correlation = np.fft.ifftshift(cross_correlation)
    cross_correlation = np.fft.ifftn(cross_correlation)

    # Crop the result to the valid bounds
    if mode == "valid":
        cross_correlation = cross_correlation[: out_shape[0], : out_shape[1]]
    elif mode == "full":
        pass
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return cross_correlation


def cross_correlate(
    image: np.ndarray,
    template: np.ndarray,
    mode: Literal["valid", "full"] = "valid",
) -> np.ndarray:
    """Cross-correlates an image and template which are provided in real space.
    The result is the cross-correlation of the two images.

    Args:
        image (np.ndarray): The image.
        template (np.ndarray): The template.
        mode (Literal["valid", "full"]): The mode of the cross-correlation.
            Default is 'valid'.

    Returns:
        np.ndarray: The cross-correlation result.
    """
    out_shape = np.array(image.shape) - np.array(template.shape) + 1

    # FFT pad and cross-correlate in Fourier space
    template_fft = np.fft.fftn(template, s=image.shape)
    image_fft = np.fft.fftn(image)

    # cross_correlation = np.conj(image_fft) * template_fft
    cross_correlation = image_fft * np.conj(template_fft)
    # cross_correlation = np.fft.ifftshift(cross_correlation)
    cross_correlation = np.fft.ifftn(cross_correlation)

    # Crop the result to the valid bounds
    if mode == "valid":
        cross_correlation = cross_correlation[: out_shape[0], : out_shape[1]]
    elif mode == "full":
        pass
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return cross_correlation


def batched_cross_correlate(
    images: torch.Tensor,
    templates: torch.Tensor,
    dim: tuple[int, ...],
    mode: Literal["valid", "full"] = "valid",
) -> torch.Tensor:
    """Run batched cross-correlation along the specified dimensions.

    Note that accepted images and templates are assumed to be in real-space,
    and the image dimensions are assumed to be larger than the template
    dimensions along the correlation axes.

    TODO: provide functionality for passing rffts rather than real-space images

    Parameters
    ----------
    images : torch.Tensor
        The images to correlate.
    templates : torch.Tensor
        The templates to correlate.
    dim : tuple[int]
        The dimensions along which to correlate.
    mode : str, optional
        The mode of the cross-correlation, by default "valid".
    """
    # Convert negative dimensions to corresponding positive
    dim = tuple([int(d % images.ndim) for d in dim])

    image_shape = [images.shape[i] for i in dim]
    out_shape = [
        (
            images.shape[i]
            if i not in dim
            else images.shape[i] - templates.shape[i] + 1
        )
        for i in range(images.ndim)
    ]

    # FFT pad templates to the size of the images
    templates_rfft = torch.fft.rfftn(templates, dim=dim, s=image_shape)
    images_rfft = torch.fft.rfftn(images, dim=dim)

    # Perform the cross-correlation
    cross_correlation = images_rfft * templates_rfft.conj()
    cross_correlation = torch.fft.irfftn(cross_correlation, dim=dim)

    # Crop the result to the valid bounds
    if mode == "valid":
        slices = [slice(0, out_shape[i]) for i in range(images.ndim)]
        cross_correlation = cross_correlation[slices]
    elif mode == "full":
        pass
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return cross_correlation


# TODO: Normalized cross-correlation
