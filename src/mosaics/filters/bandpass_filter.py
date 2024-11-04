from typing import Tuple

import numpy as np

from mosaics.utils import _calculate_pixel_spatial_frequency


def get_bandpass_filter(
    shape: Tuple[int, int],
    pixel_size: float,
    low_resolution_cutoff: float = None,
    high_resolution_cutoff: float = None,
    low_resolution_decay_rate: float = None,
    high_resolution_decay_rate: float = None,
) -> np.ndarray:
    """Get a bandpass filter for an image with a given shape and cutoff
    frequencies.

    Args:
        shape (tuple): The shape of the image to create the bandpass filter
            for.
        pixel_size (float): The size of a pixel in the image in physical units.
        low_resolution_cutoff (float): Low resolution cutoff frequency in
            physical units. For example, 10 Angstroms; this would set the low
            cutoff to 0.1 A^-1.
        high_resolution_cutoff (float): High resolution cutoff frequency in
            physical units. For example, 2 Angstroms; this would set the high
            cutoff to 0.5 A^-1.
        low_resolution_decay_rate (float): The decay rate for the low frequency
            cutoff, in units of squared (inverse) Angstroms. Larger values mean
            a sharper cutoff with the default None applying a hard cutoff.
        high_resolution_decay_rate (float): The decay rate for the high
            frequency cutoff, in units of squared (inverse) Angstroms. Larger
            values mean a sharper cutoff with the default None applying a hard
            cutoff.

    Returns:
        np.ndarray: The bandpass filter for the image.
    """
    if low_resolution_cutoff is None:
        low_resolution_cutoff = -np.inf
    else:
        low_resolution_cutoff = 1 / low_resolution_cutoff

    if high_resolution_cutoff is None:
        high_resolution_cutoff = np.inf
    else:
        high_resolution_cutoff = 1 / high_resolution_cutoff

    r = _calculate_pixel_spatial_frequency(shape, pixel_size=pixel_size)

    # return (r >= low_cutoff) * (r <= high_cutoff)
    if low_resolution_decay_rate is not None:
        low_pass = np.exp(
            -(low_resolution_decay_rate**2) * np.maximum(0, low_cutoff - r)
        )
    else:
        low_pass = r >= low_cutoff

    if high_resolution_decay_rate is not None:
        high_pass = np.exp(
            -(high_resolution_decay_rate**2) * np.maximum(0, r - high_cutoff)
        )
    else:
        high_pass = r <= high_cutoff

    return low_pass * high_pass
