from typing import Optional

import mrcfile
import numpy as np
import scipy as sp

from mosaics.data_structures import ContrastTransferFunction
from mosaics.filters.whitening_filter import compute_power_spectral_density_1D
from mosaics.utils import _calculate_pixel_spatial_frequency


class PowerSpectralDensity:
    """Class which represents the power spectral density of an image.
    Implements i/o functions for importing and exporting PSD data as numpy
    arrays.

    Attributes:
    -----------

    pixel_size (float): The pixel size of the image used to calculate the power
        spectral density in Angstroms.
    psd_array (np.ndarray): Intensity values of the power spectral density.
    psd_frequencies (np.ndarray): Frequencies (in units of 1/Angstroms)
        corresponding to the PSD values.

    Methods:
    --------

    from_micrograph: Create a PowerSpectralDensity object from a Micrograph
        object.
    from_numpy_txt: Create a PowerSpectralDensity object from a numpy text
        file.
    to_numpy_txt: Write the PSD data to a numpy text file.
    apply_whitening_filter: Apply a whitening filter to an image using the
        held power spectral density values.

    """

    pixel_size: float
    psd_array: np.ndarray
    psd_frequencies: np.ndarray

    _header = "Row 1: Frequencies (1/Angstroms)\nRow 2: PSD values"

    @classmethod
    def from_micrograph(
        cls, micrograph: "Micrograph"
    ) -> "PowerSpectralDensity":
        """Create a PowerSpectralDensity object from a Micrograph object.

        Args:
        -----

        micrograph (Micrograph): The Micrograph object to create the PSD from.

        Returns:

        PowerSpectralDensity: A PowerSpectralDensity object created from the
            Micrograph object.
        """
        return cls(micrograph.image_array, micrograph.pixel_size)

    def __init__(self, image: np.ndarray, pixel_size: float):
        self.pixel_size = pixel_size

        # Compute the power spectral density of the image
        tmp = compute_power_spectral_density_1D(image, pixel_size)
        self.psd_array = tmp[0]
        self.psd_frequencies = tmp[1]

    def to_numpy_txt(self, path: str) -> None:
        """Write the PSD data to a numpy text file.

        Args:
        -----

        path (str): Path to the numpy text file.

        Returns:
        --------

        None
        """
        np.savetxt(
            path,
            (self.psd_frequencies, self.psd_array),
            header=self._header,
        )

    @classmethod
    def from_numpy_txt(cls, path: str) -> "PowerSpectralDensity":
        """Create a PowerSpectralDensity object from a numpy text file.

        Args:
        -----

        path (str): Path to the numpy text file.

        Returns:
        --------

        PowerSpectralDensity: A PowerSpectralDensity object created from the
            data in the numpy text file.
        """
        data = np.loadtxt(path)
        return cls(data[1], data[0])

    def apply_whitening_filter(
        self, image: np.ndarray, image_pixel_size: float = None
    ) -> np.ndarray:
        """Apply a whitening filter to an image using the held power spectral
        density values.

        Args:
        -----

        image (np.ndarray): The image to whiten.
        image_pixel_size (float): The pixel size of the image in Angstroms.
            If not provided, the pixel size of the held PSD object is used.

        Returns:
        --------

        np.ndarray: The whitened image.

        """
        if image_pixel_size is None:
            image_pixel_size = self.pixel_size

        freq_grid = _calculate_pixel_spatial_frequency(
            image.shape, image_pixel_size
        )
        freq_grid = freq_grid.flatten()

        # Interpolate the PSD values to the image's spatial frequencies
        psd_interp = sp.interpolate.interp1d(
            points=[self.psd_frequencies],
            values=[self.psd_array],
            xi=freq_grid,
            method="linear",
            bounds_error=False,
            fill_value=1e-10,
        )

        # Reshape and apply the filter to the image
        psd_interp = psd_interp.reshape(image.shape)
        np.where(psd_interp == 0, 1e12, psd_interp)
        whitening_filter = 1 / psd_interp

        image_fft = np.fft.fft2(image)
        image_fft = np.fft.fftshift(image_fft)
        image_fft *= whitening_filter
        image_fft = np.fft.ifftshift(image_fft)
        image_whitened = np.fft.ifft2(image_fft)

        return np.real(image_whitened)


class Micrograph:
    """Class for handling micrograph data and common operations.

    Attributes: TODO

    Methods: TODO

    """

    pixel_size: float  # In Angstroms, assume square pixels
    image_array: np.ndarray
    image_array_whitened: Optional[np.ndarray] = None

    image_path: Optional[str] = None
    image_whitened_path: Optional[str] = None

    contrast_transfer_function: Optional[ContrastTransferFunction] = None
    power_spectral_density: Optional[PowerSpectralDensity] = None

    @classmethod
    def from_mrc(
        cls,
        mrc_path: str,
        contrast_transfer_function: ContrastTransferFunction = None,
        calculate_power_spectral_density: bool = True,
    ) -> "Micrograph":
        """Create a Micrograph object from an MRC file."""
        with mrcfile.open(mrc_path) as mrc:
            image_array = mrc.data.squeeze().copy()
            pixel_size = mrc.voxel_size.x

        return cls(
            image_array,
            pixel_size,
            mrc_path,
            contrast_transfer_function,
            calculate_power_spectral_density,
        )

    def __init__(
        self,
        image_array: np.ndarray,
        pixel_size: float,
        image_path: str = None,
        contrast_transfer_function: ContrastTransferFunction = None,
        calculate_power_spectral_density: bool = True,
    ):
        self.image_array = image_array
        self.pixel_size = pixel_size
        self.image_path = image_path
        self.contrast_transfer_function = contrast_transfer_function

        if calculate_power_spectral_density:
            self.power_spectral_density = PowerSpectralDensity(
                image_array, pixel_size
            )

    def whiten_image(self) -> np.ndarray:
        """Helper function to whiten the passed image using the held
        PowerSpectralDensity object. PSD is calculated if not already done.
        Updates the held `image_array_whitened` attribute and returns the
        whitened image.

        Args:
        -----
        None

        Returns:
        --------

            np.ndarray: The whitened image.
        """
        if self.power_spectral_density is None:
            self.power_spectral_density = PowerSpectralDensity(
                self.image_array, self.pixel_size
            )

        tmp = self.power_spectral_density.apply_whitening_filter(
            self.image_array, self.pixel_size
        )
        self.image_array_whitened = tmp

        return tmp

    # def _validate_to_particle_stack_inputs(
    #     self,
    #     positions_x: np.ndarray,
    #     positions_y: np.ndarray,
    #     particle_orientations: np.ndarray = None,
    #     particle_defocus_parameters: np.ndarray = None,
    #     particle_z_scores: np.ndarray = None,
    #     particle_mip_values: np.ndarray = None,
    # ) -> None:
    #     """Helper function to validate inputs for particle extraction.

    #     Checks that the number of particles is consistent across all provided
    #     arrays and that the shapes of the arrays match what is expected.

    #     Args:
    #         particle_positions: Array of particle positions, shape (N,2)
    #         particle_orientations: Optional array of orientations, (N,3)
    #         particle_defocus_parameters: Optional array of CTF params, shape
    #             (N,3)
    #         particle_z_scores: Optional array of z-scores, shape (N,)
    #         particle_mip_values: Optional array of MIP values, shape (N,)
    #     """
    #     assert (
    #         positions_x.shape == positions_y.shape
    #     ), "Positions x and y must have the same shape."

    #     # Same length validation
    #     if particle_orientations is not None:
    #         assert (
    #             positions_x.shape[0] == particle_orientations.shape[0]
    #         ), "Number of particle positions and orientations must be equal."

    #     if particle_defocus_parameters is not None:
    #         assert (
    #             positions_x.shape[0] == particle_defocus_parameters.shape[0]
    #         ), "Num of particle positions and CTF parameters must be equal."

    #     if particle_z_scores is not None:
    #         assert (
    #             positions_x.shape[0] == particle_z_scores.shape[0]
    #         ), "Number of particle positions and z-scores must be equal."

    #     if particle_mip_values is not None:
    #         assert (
    #             positions_x.shape[0] == particle_mip_values.shape[0]
    #         ), "Number of particle positions and MIP values must be equal."

    #     # Shape validation
    #     if particle_orientations is not None:
    #         assert (
    #             particle_orientations.shape[1] == 3
    #         ), "Orientation array must have 3 columns."

    #     if particle_defocus_parameters is not None:
    #         assert (
    #             particle_defocus_parameters.shape[1] == 3
    #         ), "Defocus parameter array must have 3 columns."

    #     if particle_z_scores is not None:
    #         assert (
    #             particle_z_scores.ndim == 1
    #         ), "Z-score array must be 1-dimensional."

    #     if particle_mip_values is not None:
    #         assert (
    #             particle_mip_values.ndim == 1
    #         ), "MIP value array must be 1-dimensional."

    # def to_particle_stack(
    #     self,
    #     box_size: tuple[int, int],
    #     positions_x: np.ndarray,
    #     positions_y: np.ndarray,
    #     positions_reference: Literal["center", "corner"] = "center",
    #     handle_bounds: Literal["crop", "fill", "error"] = "error",
    #     use_whitened_image: bool = False,
    #     particle_orientations: np.ndarray = None,
    #     particle_defocus_parameters: np.ndarray = None,
    #     particle_z_scores: np.ndarray = None,
    #     particle_mip_values: np.ndarray = None,
    # ) -> "ParticleStack":
    #     """Extract particles from the micrograph using the provided particle
    #     positions and other optional information about each particle.

    #     Args:
    #     -----

    #     TODO: Args

    #     Returns:
    #     --------

    #         ParticleStack: A ParticleStack object containing the extracted
    #             particles.

    #     """
    #     assert (
    #         self.contrast_transfer_function is not None
    #     ), "Currently, contrast_transfer_function must not be none."

    #     self._validate_to_particle_stack_inputs(
    #         positions_x,
    #         positions_y,
    #         particle_orientations,
    #         particle_defocus_parameters,
    #         particle_z_scores,
    #         particle_mip_values,
    #     )

    #     ref_image = (
    #         self.image_array_whitened
    #         if use_whitened_image
    #         else self.image_array
    #     )

    #     # Iterate over each position and extract the particle
    #     particle_images = []
    #     for i in range(positions_x.shape[0]):
    #         particle_images.append(
    #             get_cropped_region_of_image(
    #                 ref_image,
    #                 box_size,
    #                 positions_x[i],
    #                 positions_y[i],
    #                 positions_reference,
    #                 handle_bounds,
    #             )
    #         )

    #     particle_images = np.array(particle_images)

    #     # Convert x, y positions to a single numpy array
    #     particle_positions = np.vstack((positions_x, positions_y)).T

    #     return ParticleStack(
    #         pixel_size=self.pixel_size,
    #         box_size=box_size,
    #         particle_images=particle_images,
    #         voltage=self.contrast_transfer_function.voltage,
    #         spherical_aberration=(
    #             self.contrast_transfer_function.spherical_aberration
    #         ),
    #         amplitude_contrast_ratio=(
    #             self.contrast_transfer_function.amplitude_contrast_ratio
    #         ),
    #         B_factor=self.contrast_transfer_function.B_factor,
    #         particle_positions=particle_positions,
    #         particle_index=np.arange(particle_positions.shape[0]),
    #         particle_class=None,
    #         particle_orientations=particle_orientations,
    #         particle_defocus_parameters=particle_defocus_parameters,
    #         particle_image_stack_paths=None,
    #         particle_micrograph_paths=[self.image_path] * num_particles,
    #         particle_psd_paths=None,  # TODO: Add PSD
    #     )
