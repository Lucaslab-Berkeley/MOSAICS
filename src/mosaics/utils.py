import numpy as np
import pandas as pd
from scipy.special import erf
from pathlib import Path

from typing import Tuple, Union, Literal

import json

# Taken from Table 4.3.2.2. of International Tables for Crystallography Vol. C Third edition (2004)
SCATTERING_PARAMS_PATH = Path(__file__).parent / "elastic_scattering_factors.json"

with open(SCATTERING_PARAMS_PATH, "r") as f:
    data = json.load(f)

SCATTERING_PARAMETERS_A = {k: v for k, v in data["parameters_a"].items() if v != []}
SCATTERING_PARAMETERS_B = {k: v for k, v in data["parameters_b"].items() if v != []}
SCATTERING_COEFFICIENT = 4.787764736e-19  # h^2 / (2pi m_0 e)


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
    # points: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    sigma: Union[float, np.ndarray],
    bins: Tuple[np.ndarray, np.ndarray],
    weights: np.ndarray = None,
    # density: bool = False,
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
    # Prepare some of the input arguments
    if weights is None:
        weights = np.ones_like(x)

    if isinstance(sigma, float):
        sigma = np.full_like(x, sigma)

    # Checks for expected input types
    assert len(x) == len(y), "Length of x and y must match."
    assert len(x) == len(sigma), "Length sigma must mach number of points."
    assert len(x) == len(weights), "Length of weights must match number of points."
    assert len(bins) == 2, "Bins must be a tuple of two arrays."

    # Set up 2D grid of integer points for the histogram
    dim0 = np.arange(bins[0].size)
    dim1 = np.arange(bins[1].size)
    xx, yy = np.meshgrid(dim0, dim1, indexing="ij")
    shape = xx.shape

    # Transform x and y to pixel-like coordinates
    x = (x - bins[0].min()) / (bins[0].max() - bins[0].min()) * (bins[0].size - 1)
    y = (y - bins[1].min()) / (bins[1].max() - bins[1].min()) * (bins[1].size - 1)

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


# def histogram_2d_linear_interpolation(points: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
def histogram_2d_linear_interpolation(
    x: np.ndarray,
    y: np.ndarray,
    bins: Tuple[np.ndarray, np.ndarray],
    # density: bool = False,
    weights: np.ndarray = None,
) -> np.ndarray:
    """Given a set of 2D points with associated values, interpolate the values using a
    2D linear interpolation. Points are assumed to be transformed into pixel-like
    coordinates (e.g. ranging from (0, 0) to (shape[0], shape[1])).

    Parameters:
        (np.ndarray) points: Array of 2D points with associated values.
        (Tuple[int, int]) shape: Shape of the output image.

    Returns: TODO
    """
    assert len(x) == len(y), "Length of x and y must match."

    if weights is not None:
        assert len(weights) == len(x), "Length of weights must match number of points."

    # Transform x and y to pixel-like coordinates
    x = (x - bins[0].min()) / (bins[0].max() - bins[0].min()) * (bins[0].size - 1)
    y = (y - bins[1].min()) / (bins[1].max() - bins[1].min()) * (bins[1].size - 1)

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


def _scattering_potential_single_atom_2d(
    pos: Tuple[np.ndarray, np.ndarray],  # From meshgrid
    atom_id: str,
    b_factor: float,
):
    r"""Calculate the scattering potential around some pixels in 2D space according to
    equation 10 in https://journals.iucr.org/m/issues/2021/06/00/rq5007/index.html

    .. math::
        \rho(r) = \dfrac{h^2}{2\pi m_0 e} \sum_{i=1}^{5} a_i \left(\dfrac{4\pi}{b_i + B_n}\right)^{3/2} \exp\left(\dfrac{-4\pi^2 r^2}{b_i + B_n}\right)

    where :math:`B` is the B-factor, :math:`a_i` and :math:`b_i` are fit parameters for
    the 5 Gaussians that approximate the scattering potential of the atom.

    TODO: Finish docstring
    """
    x, y = pos
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    amp = np.zeros((x.shape))

    # Iterate over each of the 5 exponential terms
    for i in range(5):
        bb = SCATTERING_PARAMETERS_B[atom_id][i] + b_factor
        
        # NOTE: Does not include [h^2 / (2pi m_0 e)] factor
        # NOTE: This reduces the intensity of perineal atoms slightly when compared to cisTEM
        coefficient = SCATTERING_PARAMETERS_A[atom_id][i]
        coefficient *= (4 * np.pi / bb) ** 1.5
        exp_term = np.exp(-4 * np.pi * np.pi * (x**2 + y**2) / bb)
        
        amp += coefficient * exp_term
    
    return amp


def calculate_scattering_potential_2d(
    x: np.ndarray,  # in Angstroms
    y: np.ndarray,  # in Angstroms
    atom_ids: np.ndarray,
    b_factors: np.ndarray,
    bins: Tuple[np.ndarray, np.ndarray],  # coordinates in Angstroms
    alpha: float = 0.01,
):
    """TODO: docstring"""
    cutoff_pixels = 12  # TODO: Calculate based on maximum B-factor and pixel size

    shape = (bins[0].size, bins[1].size)

    # Reduced number of pixels to calculate the scattering potential around
    pos0 = np.arange(cutoff_pixels * 2 + 1).astype(np.float32) - cutoff_pixels
    pos1 = np.arange(cutoff_pixels * 2 + 1).astype(np.float32) - cutoff_pixels
    pos = np.meshgrid(pos0, pos1)

    # Transform x and y to pixel-like coordinates
    x = (x - bins[0].min()) / (bins[0].max() - bins[0].min()) * (bins[0].size - 1)
    y = (y - bins[1].min()) / (bins[1].max() - bins[1].min()) * (bins[1].size - 1)

    # Calculate the scattering potential for each atom
    histogram = np.zeros((bins[0].size, bins[1].size))
    for i in range(len(atom_ids)):
        x_int = np.round(x[i]).astype(np.int32)
        y_int = np.round(y[i]).astype(np.int32)

        # Offset positions in the square around the atom
        dx = x[i] - x_int
        dy = y[i] - y_int
        tmp_pos = [pos[0] - dx, pos[1] - dy]

        # Determine which pixel positions to calculate at and where to add the density
        # TODO: Clean up this indexing to accelerate
        kernel_window = np.s_[
            max(x_int - cutoff_pixels, 0) : min(x_int + cutoff_pixels + 1, shape[0]),
            max(y_int - cutoff_pixels, 0) : min(y_int + cutoff_pixels + 1, shape[1]),
        ]

        tmp_pos[0] = tmp_pos[0][
            cutoff_pixels
            - min(x_int, cutoff_pixels) : cutoff_pixels
            + max(shape[0] - x_int - 1, cutoff_pixels)
            + 1,
            cutoff_pixels
            - min(y_int, cutoff_pixels) : cutoff_pixels
            + max(shape[1] - y_int - 1, cutoff_pixels)
            + 1,
        ]
        tmp_pos[1] = tmp_pos[1][
            cutoff_pixels
            - min(x_int, cutoff_pixels) : cutoff_pixels
            + max(shape[0] - x_int - 1, cutoff_pixels)
            + 1,
            cutoff_pixels
            - min(y_int, cutoff_pixels) : cutoff_pixels
            + max(shape[1] - y_int - 1, cutoff_pixels)
            + 1,
        ]

        histogram[kernel_window] += _scattering_potential_single_atom_2d(
            tmp_pos,
            atom_ids[i],
            b_factors[i],
        )

    return histogram

def get_cropped_region_of_image(
    image: np.ndarray,
    box_size: Tuple[int, int],
    positions_x: int,
    positions_y: int,
    positions_reference: Literal["center", "corner"] = "center",
    handle_bounds: Literal["crop", "fill", "error"] = "error",
) -> np.ndarray:
    """Crop the region with given box size and position out of an image. Handles
    position references and bounds checking.
    
    TODO: Finish docstring
    """
    # Handle the position reference
    if positions_reference == "center":
        positions_x = int(positions_x - box_size[1] / 2)
        positions_y = int(positions_y - box_size[0] / 2)

    x_bounds = [positions_x, positions_x + box_size[1]]
    y_bounds = [positions_y, positions_y + box_size[0]]
    
    # Handle the bounds checking
    _bounds_flag = False
    if (
        x_bounds[0] < 0 or x_bounds[1] > image.shape[1] or
        y_bounds[0] < 0 or y_bounds[1] > image.shape[0]
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
                
    tmp_image = image[y_bounds[0]:y_bounds[1], x_bounds[0]:x_bounds[1]]

    if _bounds_flag and handle_bounds == "fill":
        tmp_image = np.pad(
            tmp_image,
            ((y_clips[0], y_clips[1]), (x_clips[0], x_clips[1])),
            mode="constant",
            constant_values=np.mean(tmp_image),  # TODO: Options for fill?
        )

    return tmp_image

def parse_out_coordinates_result(filename) -> pd.DataFrame:
    """Parse the columns of the make_template_result out_coordinates.txt file and place
    all the columns into a pandas DataFrame. First row defines the column names,
    separated by whitespace, with the subsequent rows in the file being the data.

    Arguments:
        (str) filename: The path to the out_coordinates.txt file to parse

    Returns:
        (pd.DataFrame) df: The DataFrame containing the parsed data
    """
    # Get the column names from the first comment line
    with open(filename, "r") as f:
        first_line = f.readline()
    column_names = first_line.strip().split()[1:]  # First character is a comment

    coord_df = pd.read_csv(filename, sep=r"\s+", skiprows=1, names=column_names)

    return coord_df
