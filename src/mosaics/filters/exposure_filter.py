from typing import Tuple

import numpy as np

from mosaics.utils import _calculate_pixel_spatial_frequency

# Fit constants from eq. (3) in https://doi.org/10.7554/eLife.06980
# for optimal exposure
FIT_CONSTANT_A = 0.245
FIT_CONSTANT_B = -1.665
FIT_CONSTANT_C = 2.81


def calculate_optimal_exposure(
    k: float,
    a: float = FIT_CONSTANT_A,
    b: float = FIT_CONSTANT_B,
    c: float = FIT_CONSTANT_C,
) -> float:
    """Calculate the optimal exposure for a given spatial frequency, k, using
    the fitted equation Ne(k) = a * k^b + c

    Args:
        k (float): The spatial frequency in A^-1.
        a (float): Equation parameter
        b (float): Equation parameter
        c (float): Equation parameter

    Returns:
        float: The optimal exposure, in units of e-/A^2
    """
    k = np.maximum(k, 1e-8)  # Avoid division by zero error for zero-frequency

    return a * np.power(k, b) + c


# def calculate_exposure_filter(
#     shape: Tuple[int, int],
#     pixel_size: float,
#     exposure: float,
# ):
#     """For an 2D image with a given shape and pixel size (in A), calculate
#     the exposure filter for a single exposure (in e-/A^2).

#     Args:
#         shape (tuple): The shape of the image to create the exposure filter
#           for.
#         pixel_size (float): The size of a pixel in the image in physical
#           units.
#         exposure (float): The exposure of the image in e-/A^2.

#     Returns:
#         np.ndarray: A 2D array for the exposure filter.
#     """
#     freq_img = _calculate_pixel_spatial_frequency(
#          shape, pixel_size=pixel_size
#      )
#     opt_exposure = calculate_optimal_exposure(freq_img)

#     raise NotImplementedError


def calculate_cumulative_exposure_filter(
    shape: Tuple[int, int],
    pixel_size: float,
    exposure_start: float,
    exposure_end: float,
    # TODO: Decide how to handle various fit constants or other exposure models
) -> np.ndarray:
    """For an 2D image with a given shape and pixel size (in A), calculate the
    cumulative exposure filter from the beginning and end exposure (in e-/A^2).

    Args:
        shape (tuple): The shape of the image to create the exposure filter
            for.
        pixel_size (float): The size of a pixel in the image in physical units.
        exposure_start (float): The starting exposure in e-/A^2.
        exposure_end (float): The ending exposure in e-/A^2.

    Returns:
        np.ndarray: A 2D array for the cumulative exposure filter.
    """
    freq_img = _calculate_pixel_spatial_frequency(shape, pixel_size=pixel_size)
    opt_exposure = calculate_optimal_exposure(freq_img)

    # Analytical expression for the cumulative exposure
    exposure_filter = 2 * opt_exposure
    exposure_filter *= np.exp(-exposure_start / (2 * opt_exposure)) - np.exp(
        -exposure_end / (2 * opt_exposure)
    )

    return exposure_filter / (exposure_end - exposure_start)


def apply_cumulative_exposure_filter(
    image: np.ndarray,
    pixel_size: float,
    exposure_start: float,
    exposure_end: float,
) -> np.ndarray:
    """Helper function to apply a cumulative exposure filter to an image and
        return the real-space filtered image.

    Args:
        image (np.ndarray): The image to apply the filter to.
        pixel_size (float): The size of a pixel in the image in physical units.
        exposure_start (float): The starting exposure in e-/A^2.
        exposure_end (float): The ending exposure in e-/A^2.

    Returns:
        np.ndarray: The filtered image.
    """
    exposure_filter = calculate_cumulative_exposure_filter(
        image.shape, pixel_size, exposure_start, exposure_end
    )
    exposure_filter = np.fft.ifftshift(exposure_filter)

    # Apply the filter in Fourier space
    image_fft = np.fft.fft2(image)
    image_fft *= exposure_filter
    image_filtered = np.fft.ifft2(image_fft)

    return np.real(image_filtered)
