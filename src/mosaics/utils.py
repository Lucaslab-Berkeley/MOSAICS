from typing import Literal
from typing import Union

import numpy as np
import pandas as pd


def _calculate_pixel_radial_distance(shape: tuple[int, int]) -> np.ndarray:
    """Helper function for getting a radial distance map for an image with a
    given shape. Position for which the radial distance is calculated is
    assumed at the center of the image.
    """
    x = np.arange(shape[1]) - shape[1] / 2
    y = np.arange(shape[0]) - shape[0] / 2
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)

    return r


def _calculate_pixel_spatial_frequency(
    shape: tuple[int, int], pixel_size: float
) -> np.ndarray:
    """Helper function for getting a spatial frequency map for an image with a
    given shape and pixel size. Position for which the spatial frequency is
    calculated is assumed at the center of the image.
    """
    x = np.fft.fftfreq(shape[1], d=pixel_size)
    y = np.fft.fftfreq(shape[0], d=pixel_size)
    xx, yy = np.meshgrid(x, y)
    freq = np.sqrt(xx**2 + yy**2)

    freq = np.fft.fftshift(freq)

    return freq


def _calculate_pixel_frequency_angle(
    shape: tuple[int, int], pixel_size: float
) -> np.ndarray:
    """TODO: Docstring"""
    x = np.fft.fftfreq(shape[1], d=pixel_size)
    y = np.fft.fftfreq(shape[0], d=pixel_size)
    xx, yy = np.meshgrid(x, y)
    angle = np.arctan2(yy, xx)

    angle = np.fft.fftshift(angle)

    return angle


def _gaussian_kernel_distance_cutoff(
    sigma: float, alpha: float = 0.01
) -> float:
    """Given an isotropic 2D Gaussian with standard deviation sigma, in units
    of pixels, this function returns the radius of pixels to consider for the
    kernel. The cutoff value, alpha, is relative to the maximum value of the
    Gaussian.

    Parameters:
        (float) sigma: Standard deviation of the Gaussian kernel, in units of
            pixels.
        (float) alpha: Relative cutoff value for the Gaussian kernel.

    Returns:
        (float): The distance cutoff, in pixels, for the Gaussian kernel.
    """
    d = sigma * np.sqrt(-2 * np.log(alpha))

    return d


def histogram_2d_gaussian_interpolation(
    # points: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    sigma: Union[float, np.ndarray],
    bins: tuple[np.ndarray, np.ndarray],
    weights: np.ndarray = None,
    # density: bool = False,
    alpha: float = 0.01,
) -> np.ndarray:
    """Given a set of 2D points with associated values, interpolate the values
    using a 2D Gaussian kernel. Points are assumed to be transformed into
    pixel-like coordinates (e.g. ranging from (0, 0) to (shape[0], shape[1])).


    NOTE: The current implementation requires the kernel be isotropic with the
    same standard deviation for all points.

    Parameters:
        (np.ndarray) points: Array of 2D points with associated values.
        (float) sigma: Standard deviation of the Gaussian kernel, in units of
            pixels.
        (float) alpha: Relative cutoff value for the Gaussian kernel.
        (tuple[int, int]) shape: Shape of the output image.

    Returns:
    """
    # Prepare some of the input arguments
    if weights is None:
        weights = np.ones_like(x)

    if isinstance(sigma, float):
        sigma = np.full_like(x, sigma)

    # Checks for expected input types
    assert len(x) == len(y), "Length of x and y must match."
    assert len(x) == len(sigma), "Length sigma must mach number of points."
    assert len(x) == len(
        weights
    ), "Length of weights must match number of points."
    assert len(bins) == 2, "Bins must be a tuple of two arrays."

    # Set up 2D grid of integer points for the histogram
    dim0 = np.arange(bins[0].size)
    dim1 = np.arange(bins[1].size)
    xx, yy = np.meshgrid(dim0, dim1, indexing="ij")
    shape = xx.shape

    # Transform x and y to pixel-like coordinates
    x = (
        (x - bins[0].min())
        / (bins[0].max() - bins[0].min())
        * (bins[0].size - 1)
    )
    y = (
        (y - bins[1].min())
        / (bins[1].max() - bins[1].min())
        * (bins[1].size - 1)
    )

    # Based on sigma, find distance of points to consider
    d_cutoff = int(_gaussian_kernel_distance_cutoff(sigma.max(), alpha) + 1)

    histogram = np.zeros(shape, dtype=np.float32)
    for _x, _y, _s in zip(x, y, sigma):
        x_int = int(np.round(_x))
        y_int = int(np.round(_y))

        # Only calculate density where kernel has significant density
        # Also handles handling points near edge of image
        kernel_window = np.s_[
            max(x_int - d_cutoff, 0) : min(x_int + d_cutoff + 1, shape[0]),
            max(y_int - d_cutoff, 0) : min(y_int + d_cutoff + 1, shape[1]),
        ]

        histogram[kernel_window] += np.exp(
            -((xx[kernel_window] - _x) ** 2 + (yy[kernel_window] - _y) ** 2)
            / (2 * _s**2)
        )

    return histogram


# def histogram_2d_linear_interpolation(
#     points: np.ndarray, shape: tuple[int, int]
# ) -> np.ndarray:
def histogram_2d_linear_interpolation(
    x: np.ndarray,
    y: np.ndarray,
    bins: tuple[np.ndarray, np.ndarray],
    # density: bool = False,
    weights: np.ndarray = None,
) -> np.ndarray:
    """Given a set of 2D points with associated values, interpolate the values
    using a 2D linear interpolation. Points are assumed to be transformed into
    pixel-like coordinates (e.g. ranging from (0, 0) to (shape[0], shape[1])).

    Parameters:
        (np.ndarray) points: Array of 2D points with associated values.
        (tuple[int, int]) shape: Shape of the output image.

    Returns: TODO
    """
    assert len(x) == len(y), "Length of x and y must match."

    if weights is not None:
        assert len(weights) == len(
            x
        ), "Length of weights must match number of points."

    # Transform x and y to pixel-like coordinates
    x = (
        (x - bins[0].min())
        / (bins[0].max() - bins[0].min())
        * (bins[0].size - 1)
    )
    y = (
        (y - bins[1].min())
        / (bins[1].max() - bins[1].min())
        * (bins[1].size - 1)
    )

    histogram = np.zeros((bins[0].size, bins[1].size), dtype=np.float32)

    for _x, _y in zip(x, y):
        i_floor = int(np.floor(_x))
        j_floor = int(np.floor(_y))
        i_ceil = i_floor + 1
        j_ceil = j_floor + 1

        dx = _x - i_floor
        dy = _y - j_floor

        histogram[i_floor, j_floor] += (1 - dx) * (1 - dy)
        histogram[i_floor, j_ceil] += (1 - dx) * dy
        histogram[i_ceil, j_floor] += dx * (1 - dy)
        histogram[i_ceil, j_ceil] += dx * dy

    # TODO: Implement density option

    return histogram


def get_cropped_region_of_image(
    image: np.ndarray,
    box_size: tuple[int, int],
    pos_x: int,
    pos_y: int,
    positions_reference: Literal["center", "top_left"] = "center",
    handle_bounds: Literal["crop", "fill", "error"] = "error",
) -> np.ndarray:
    """Crop the region with given box size and position out of an image.
    Handles position references and bounds checking.

    TODO: Finish docstring
    """
    assert positions_reference in [
        "center",
        "top_left",
    ], "positions_reference must be either 'center' or 'top_left'."

    # Handle the position reference
    if positions_reference == "center":
        pos_x = int(pos_x - box_size[1] / 2)
        pos_y = int(pos_y - box_size[0] / 2)

    x_bounds = [pos_x, pos_x + box_size[1]]
    y_bounds = [pos_y, pos_y + box_size[0]]

    # Handle the bounds checking
    _bounds_flag = False
    if (
        x_bounds[0] < 0
        or x_bounds[1] > image.shape[1]
        or y_bounds[0] < 0
        or y_bounds[1] > image.shape[0]
    ):
        if handle_bounds == "error":
            raise ValueError("Selected region is out of bounds for the image.")

        _bounds_flag = True
        x_clips = [
            max(-x_bounds[0], 0),
            max(x_bounds[1] - image.shape[1], 0),
        ]
        y_clips = [
            max(-y_bounds[0], 0),
            max(y_bounds[1] - image.shape[0], 0),
        ]

        x_bounds = [max(x_bounds[0], 0), min(x_bounds[1], image.shape[1])]
        y_bounds = [max(y_bounds[0], 0), min(y_bounds[1], image.shape[0])]

    tmp_image = image[y_bounds[0] : y_bounds[1], x_bounds[0] : x_bounds[1]]

    if _bounds_flag and handle_bounds == "fill":
        tmp_image = np.pad(
            tmp_image,
            ((y_clips[0], y_clips[1]), (x_clips[0], x_clips[1])),
            mode="constant",
            constant_values=np.mean(tmp_image),  # TODO: Options for fill?
        )

    return tmp_image


def parse_out_coordinates_result(filename) -> pd.DataFrame:
    """Parse the columns of the make_template_result out_coordinates.txt file
    and place all the columns into a pandas DataFrame. First row defines the
    column names, separated by whitespace, with the subsequent rows in the file
    being the data.

    Arguments:
        (str) filename: The path to the out_coordinates.txt file to parse

    Returns:
        (pd.DataFrame) df: The DataFrame containing the parsed data
    """
    # Get the column names from the first comment line
    with open(filename, "r") as f:
        first_line = f.readline()
    # First character is a comment
    column_names = first_line.strip().split()[1:]

    coord_df = pd.read_csv(
        filename, sep=r"\s+", skiprows=1, names=column_names
    )

    return coord_df
