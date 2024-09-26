import numpy as np

from typing import Tuple


def _calculate_pixel_radial_distance(shape: Tuple[int, int]) -> np.ndarray:
    """Helper function for getting a radial distance map for an image with a given
    shape. Position for which the radial distance is calculated is assumed at the
    center of the image.
    """
    x = np.arange(shape[1]) - shape[1] / 2
    y = np.arange(shape[0]) - shape[0] / 2
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)

    return r


def _calculate_pixel_spatial_frequency(
    shape: Tuple[int, int], pixel_size: float
) -> np.ndarray:
    """Helper function for getting a spatial frequency map for an image with a given
    shape and pixel size. Position for which the spatial frequency is calculated is
    assumed at the center of the image.
    """
    x = np.fft.fftfreq(shape[1], d=pixel_size)
    y = np.fft.fftfreq(shape[0], d=pixel_size)
    xx, yy = np.meshgrid(x, y)
    freq = np.sqrt(xx**2 + yy**2)

    freq = np.fft.fftshift(freq)

    return freq


def _calculate_pixel_frequency_angle(
    shape: Tuple[int, int], pixel_size: float
) -> np.ndarray:
    """TODO: Docstring"""
    x = np.fft.fftfreq(shape[1], d=pixel_size)
    y = np.fft.fftfreq(shape[0], d=pixel_size)
    xx, yy = np.meshgrid(x, y)
    angle = np.arctan2(yy, xx)

    angle = np.fft.fftshift(angle)

    return angle


def _gaussian_kernel_distance_cutoff(sigma: float, alpha: float = 0.01) -> float:
    """Given an isotropic 2D Gaussian with standard deviation sigma, in units of pixels,
    this function returns the radius of pixels to consider for the kernel. The cutoff
    value, alpha, is relative to the maximum value of the Gaussian.

    Parameters:
        (float) sigma: Standard deviation of the Gaussian kernel, in units of pixels.
        (float) alpha: Relative cutoff value for the Gaussian kernel.
        
    Returns:
        (float): The distance cutoff, in pixels, for the Gaussian kernel.
    """
    d = sigma * np.sqrt(-2 * np.log(alpha))

    return d


def histogram_2d_gaussian_interpolation(
    points: np.ndarray,
    sigma: float,
    shape: Tuple[int, int],
    alpha: float = 0.01,
) -> np.ndarray:
    """Given a set of 2D points with associated values, interpolate the values using a
    2D Gaussian kernel. Points are assumed to be transformed into pixel-like
    coordinates (e.g. ranging from (0, 0) to (shape[0], shape[1])).
    
    
    NOTE: The current implementation requires the kernel be isotropic with the same
    standard deviation for all points.

    Parameters:
        (np.ndarray) points: Array of 2D points with associated values.
        (float) sigma: Standard deviation of the Gaussian kernel, in units of pixels.
        (float) alpha: Relative cutoff value for the Gaussian kernel.
        (Tuple[int, int]) shape: Shape of the output image.
        
    Returns:
    """
    dim_0 = np.arange(shape[0])
    dim_1 = np.arange(shape[1])
    xx, yy = np.meshgrid(dim_0, dim_1, indexing="ij")

    # Based on sigma, find distance of points to consider
    d_cutoff = int(_gaussian_kernel_distance_cutoff(sigma, alpha) + 1)

    histogram = np.zeros(shape, dtype=np.float32)
    for i in range(points.shape[0]):
        x, y = points[i]
        x_int = int(np.round(x))
        y_int = int(np.round(y))

        # Only calculate density where kernel has significant density
        # Also handles handling points near edge of image
        kernel_window = np.s_[
            max(x_int - d_cutoff, 0) : min(x_int + d_cutoff + 1, shape[0]),
            max(y_int - d_cutoff, 0) : min(y_int + d_cutoff + 1, shape[1]),
        ]

        histogram[kernel_window] += np.exp(
            -((xx[kernel_window] - x) ** 2 + (yy[kernel_window] - y) ** 2)
            / (2 * sigma**2)
        )

    return histogram
