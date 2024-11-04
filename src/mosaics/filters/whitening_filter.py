from typing import Tuple

import numpy as np
import scipy as sp

from mosaics.utils import (_calculate_pixel_radial_distance,
                           _calculate_pixel_spatial_frequency)


def _calculate_num_psd_bins(shape: Tuple[int, int]) -> int:
    """Helper function for calculating the default number of bins to use for the radial
    averaging of the power spectral density.
    """
    n_bins = int(max(shape) / 2 + 1) * np.sqrt(2) + 1

    return int(n_bins)


def calculate_radial_sum(
    image, num_bins: int = None, interpolation: str = "linear"
) -> Tuple[np.ndarray, np.ndarray]:
    """Given a 2D image, calculate the radial sum of the image with the given number of
    bins and interpolation method. Returns the radial sum values and the bin counts.

    NOTE: For power spectral density, need to abs or square image before passing

    Args:
        image (np.ndarray): 2D image to calculate radial sum of
        num_bins (int): Number of bins to use for radial sum. If None, the number of
            bins is automatically calculated based on the image dimensions.
        interpolation (str): Interpolation method to use when calculating the radial
            sum. Currently supported options are "linear" and "nearest".
    """
    if num_bins is None:
        num_bins = _calculate_num_psd_bins(image.shape)

    r = _calculate_pixel_radial_distance(image.shape)

    # Initialize the sampling arrays
    values_sum = np.zeros(num_bins)
    counts_sum = np.zeros(num_bins)

    if interpolation == "nearest":
        indexes = np.round(r).astype(int)
        mask = np.logical_and(indexes >= 0, indexes < num_bins - 1)

        values_sum = np.bincount(indexes[mask], weights=image[mask], minlength=num_bins)
        counts_sum = np.bincount(indexes[mask], minlength=num_bins)

    elif interpolation == "linear":
        # TODO: Possibly move the common bincount routine to a separate function for
        # reduction of code duplication
        # Histogram with linear interpolation masking out-of-bounds radial values
        indexes_floor = np.floor(r).astype(int)
        weights_floor = 1 - (r - indexes_floor)
        mask = np.logical_and(indexes_floor >= 0, indexes_floor < num_bins)
        values_sum += np.bincount(
            indexes_floor[mask],
            weights=image[mask] * weights_floor[mask],
            minlength=num_bins,
        )
        counts_sum += np.bincount(
            indexes_floor[mask], weights=weights_floor[mask], minlength=num_bins
        )

        # Same hist routine as above, but for the upper indices
        indexes_ceil = np.ceil(r).astype(int)
        weights_ceil = 1 - weights_floor
        mask = np.logical_and(indexes_ceil >= 0, indexes_ceil < num_bins)
        values_sum += np.bincount(
            indexes_ceil[mask],
            weights=image[mask] * weights_ceil[mask],
            minlength=num_bins,
        )
        counts_sum += np.bincount(
            indexes_ceil[mask], weights=weights_ceil[mask], minlength=num_bins
        )

    return values_sum, counts_sum


def compute_power_spectral_density_1D(
    image, pixel_size: float = 1, is_fourier_space: bool = False, **kwargs
):
    """Given a 2D image, compute the 1D power spectral density of the image. Additional
    keyword arguments are passed to the calculate_radial_sum function.

    Args:
        image (np.ndarray): 2D image to calculate the power spectral density of
        num_bins (int): Number of bins to use for the radial sum. If None, the number of
            bins is automatically calculated based on the image dimensions.
        in_fourier_space (bool): If True, the image is assumed to already be in Fourier
            space. Default is False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The first array is the density values, and the
            second array are the frequency values.
    """
    if not is_fourier_space:
        image = np.fft.fft2(image)
        image = np.fft.fftshift(image)

    image = np.abs(image)

    # Calculate the radial sum of the image and get the PSD by normalization
    radial_sum, counts_sum = calculate_radial_sum(image, **kwargs)
    counts_sum[counts_sum == 0] = 1
    power_spectral_density = radial_sum / counts_sum

    # Figure out the frequency values associated with the bins
    num_bins = power_spectral_density.size
    max_freq = np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2) / 2  # corner pixel
    frequency_values = np.linspace(0, max_freq, num_bins) / pixel_size

    return power_spectral_density, frequency_values


def compute_power_spectral_density_2D(
    image, pixel_size: float = 1, is_fourier_space: bool = False, **kwargs
):
    """Calculates the power spectral density but maps back the spectral density into 2D
    space using linear interpolation.
    """
    if not is_fourier_space:
        image = np.fft.fft2(image)
        image = np.fft.fftshift(image)

    image = np.abs(image)

    # Calculate the radial sum of the image and get the PSD by normalization
    radial_sum, counts_sum = calculate_radial_sum(image, **kwargs)
    counts_sum[counts_sum == 0] = 1
    power_spectral_density = radial_sum / counts_sum

    r = _calculate_pixel_radial_distance(image.shape)
    r = r.flatten()

    # Use linear interpolation to map the PSD back to 2D space
    psd_image = sp.interpolate.interpn(
        points=[np.arange(power_spectral_density.size)],
        values=power_spectral_density,
        xi=r,
        method="linear",
        bounds_error=True,
        fill_value=1e-10,
    )

    psd_image = psd_image.reshape(image.shape)

    return psd_image


def get_whitening_filter(
    image, pixel_size: float = 1, is_fourier_space: bool = False, **kwargs
) -> np.ndarray:
    """TODO: Docstring"""
    power_spectrum_2D = compute_power_spectral_density_2D(
        image=image, pixel_size=pixel_size, is_fourier_space=is_fourier_space, **kwargs
    )

    # whitening_filter = 1 / np.sqrt(power_spectrum_2D)
    whitening_filter = 1 / power_spectrum_2D
    whitening_filter /= whitening_filter.max()

    return whitening_filter
