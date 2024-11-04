import json
from pathlib import Path
from typing import Tuple

import numpy as np

# Taken from Table 4.3.2.2. of International Tables for Crystallography Vol. C Third edition (2004)
SCATTERING_PARAMS_PATH = Path(__file__).parent / "elastic_scattering_factors.json"

with open(SCATTERING_PARAMS_PATH, "r") as f:
    data = json.load(f)

SCATTERING_PARAMETERS_A = {k: v for k, v in data["parameters_a"].items() if v != []}
SCATTERING_PARAMETERS_B = {k: v for k, v in data["parameters_b"].items() if v != []}
SCATTERING_COEFFICIENT = 4.787764736e-19  # h^2 / (2pi m_0 e)


def _scattering_potential_single_atom_3d(
    pos: Tuple[np.ndarray, np.ndarray, np.ndarray],  # From meshgrid
    atom_id: str,
    b_factor: float,
):
    r"""Calculate the scattering potential around some pixels in 3D space according to
    equation 10 in https://journals.iucr.org/m/issues/2021/06/00/rq5007/index.html.

    TODO: Finish docstring

    TODO: Include equation

    """
    x, y, z = pos

    amp = np.zeros((x.shape))

    # Iterate over each of the 5 exponential terms
    for i in range(5):
        bb = SCATTERING_PARAMETERS_B[atom_id][i] + b_factor

        # NOTE: Does not include [h^2 / (2pi m_0 e)] factor
        coefficient = SCATTERING_PARAMETERS_A[atom_id][i]
        coefficient *= (4 * np.pi / bb) ** 1.5
        exp_term = np.exp(-4 * np.pi * np.pi * (x**2 + y**2 + z**2) / bb)

        amp += coefficient * exp_term

    return amp


def _scattering_potential_single_atom_2d(
    pos: Tuple[np.ndarray, np.ndarray],  # From meshgrid
    atom_id: str,
    b_factor: float,
):
    r"""Calculate the scattering potential around some pixels in 2D space according to
    equation 10 in https://journals.iucr.org/m/issues/2021/06/00/rq5007/index.html,
    integrated over the z (projection) axis.

    .. math::
        \rho(x, y) = \dfrac{h^2}{2\pi m_0 e} \sum_{i=1}^{5} a_i \left(\dfrac{4\pi}{b_i + B_n}\right) \exp\left(\dfrac{-4\pi^2 (x^2 + y^2)}{b_i + B_n}\right)

    where :math:`B_n` is the B-factor, :math:`a_i` and :math:`b_i` are fit parameters for
    the 5 Gaussians that approximate the scattering potential of the atom.

    TODO: Finish docstring
    """
    x, y = pos

    amp = np.zeros((x.shape))

    # Iterate over each of the 5 exponential terms
    for i in range(5):
        bb = SCATTERING_PARAMETERS_B[atom_id][i] + b_factor

        # NOTE: Does not include [h^2 / (2pi m_0 e)] factor
        coefficient = SCATTERING_PARAMETERS_A[atom_id][i]
        coefficient *= 4 * np.pi / bb
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
    histogram = np.zeros(shape)
    for i in range(len(atom_ids)):
        x_int = np.round(x[i]).astype(np.int32)
        y_int = np.round(y[i]).astype(np.int32)

        # Offset positions in the square around the atom
        dx = x[i] - x_int
        dy = y[i] - y_int
        tmp_pos = [pos[0] - dx, pos[1] - dy]

        # Determine which pixel positions to calculate at and where to add the density
        kernel_window = np.s_[
            max(x_int - cutoff_pixels, 0) : min(x_int + cutoff_pixels + 1, shape[0]),
            max(y_int - cutoff_pixels, 0) : min(y_int + cutoff_pixels + 1, shape[1]),
        ]

        # TODO: Clean up this indexing to accelerate computation. Necessary for when
        # an atom may be close to the edge of the grid.
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


def calcualte_scattering_potential_3d(
    x: np.ndarray,  # in Angstroms
    y: np.ndarray,  # in Angstroms
    z: np.ndarray,  # in Angstroms
    atom_ids: np.ndarray,
    b_factors: np.ndarray,
    bins: Tuple[np.ndarray, np.ndarray, np.ndarray],  # coordinates in Angstroms
    alpha: float = 0.01,
) -> np.ndarray:
    """TODO: docstring"""
    cutoff_pixels = 16  # TODO: Calculate based on maximum B-factor and pixel size

    shape = (bins[0].size, bins[1].size, bins[2].size)

    # Reduced number of pixels to calculate the scattering potential around
    pos0 = np.arange(cutoff_pixels * 2 + 1).astype(np.float32) - cutoff_pixels
    pos1 = np.arange(cutoff_pixels * 2 + 1).astype(np.float32) - cutoff_pixels
    pos2 = np.arange(cutoff_pixels * 2 + 1).astype(np.float32) - cutoff_pixels
    pos = np.meshgrid(pos0, pos1, pos2)

    # Transform x, y, and z to pixel-like coordinates
    x = (x - bins[0].min()) / (bins[0].max() - bins[0].min()) * (bins[0].size - 1)
    y = (y - bins[1].min()) / (bins[1].max() - bins[1].min()) * (bins[1].size - 1)
    z = (z - bins[2].min()) / (bins[2].max() - bins[2].min()) * (bins[2].size - 1)

    histogram = np.zeros(shape)
    for i in range(len(atom_ids)):
        x_int = np.round(x[i]).astype(np.int32)
        y_int = np.round(y[i]).astype(np.int32)
        z_int = np.round(z[i]).astype(np.int32)

        # Offset positions in the cube around the atom
        dx = x[i] - x_int
        dy = y[i] - y_int
        dz = z[i] - z_int
        tmp_pos = [pos[0] - dx, pos[1] - dy, pos[2] - dz]

        # Determine which voxel positions to calculate at and where to add the density
        # TODO: Clean up this indexing to accelerate
        kernel_window = np.s_[
            max(x_int - cutoff_pixels, 0) : min(x_int + cutoff_pixels + 1, shape[0]),
            max(y_int - cutoff_pixels, 0) : min(y_int + cutoff_pixels + 1, shape[1]),
            max(z_int - cutoff_pixels, 0) : min(z_int + cutoff_pixels + 1, shape[2]),
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
            cutoff_pixels
            - min(z_int, cutoff_pixels) : cutoff_pixels
            + max(shape[2] - z_int - 1, cutoff_pixels)
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
            cutoff_pixels
            - min(z_int, cutoff_pixels) : cutoff_pixels
            + max(shape[2] - z_int - 1, cutoff_pixels)
            + 1,
        ]
        tmp_pos[2] = tmp_pos[2][
            cutoff_pixels
            - min(x_int, cutoff_pixels) : cutoff_pixels
            + max(shape[0] - x_int - 1, cutoff_pixels)
            + 1,
            cutoff_pixels
            - min(y_int, cutoff_pixels) : cutoff_pixels
            + max(shape[1] - y_int - 1, cutoff_pixels)
            + 1,
            cutoff_pixels
            - min(z_int, cutoff_pixels) : cutoff_pixels
            + max(shape[2] - z_int - 1, cutoff_pixels)
            + 1,
        ]

        histogram[kernel_window] += _scattering_potential_single_atom_3d(
            tmp_pos,
            atom_ids[i],
            b_factors[i],
        )
    return histogram
