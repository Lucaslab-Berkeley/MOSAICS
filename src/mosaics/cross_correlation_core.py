from typing import Literal

import numpy as np


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
    image_fft = np.fft.fftn(image)
    image_fft = np.fft.fftshift(image_fft)
    template_fft = np.fft.fftn(template)
    template_fft = np.fft.fftshift(template_fft)

    return cross_correlate_from_fft(image_fft, template_fft, mode)


# TODO: Normalized cross-correlation
