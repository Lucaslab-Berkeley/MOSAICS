import numpy as np
import mrcfile

from typing import Tuple
from typing import Literal

from .particle_stack import ParticleStack
from .filters.bandpass_filter import get_bandpass_filter
from .filters.whitening_filter import get_whitening_filter


class Micrograph:
    """Class for handling micrograph data and common operations.

    Attributes: TODO

    Methods: TODO

    """

    image_array: np.ndarray
    pixel_size: float  # In Angstroms, assume square pixels
    image_path: str

    ctf: ContrastTransferFunction = None

    # image_array_fft: np.ndarray
    # whitening_filter: np.ndarray = None  # for holding pre-computation
    # power_spectral_density: np.ndarray = None

    # _is_whitened: bool = False

    def from_mrc(cls, mrc_path: str):
        """Create a Micrograph object from an MRC file."""
        with mrcfile.open(mrc_path) as mrc:
            image_array = mrc.data.copy()
            pixel_size = mrc.voxel_size.x

        return cls(image_array, pixel_size, mrc_path)

    def __init__(
        self, image_array: np.ndarray, pixel_size: float, image_path: str = None
    ):
        self.image_array = image_array
        self.pixel_size = pixel_size
        self.image_path = image_path

        # # Pre-compute the FFT of the image
        # self.image_array_fft = np.fft.fft2(image_array)
        # self.image_array_fft = np.fft.fftshift(self.image_array_fft)

    # def whiten_image(self) -> None:
    #     """Apply a whitening filter to the image (in Fourier space) and recompute the
    #     real-space image. All updates are handled internally meaning the method returns
    #     None.
    #     """
    #     if self._is_whitened:
    #         return

    #     self.whitening_filter = get_whitening_filter(self.image_array, self.pixel_size)

    #     # Re-compute the image
    #     # filter applied twice to image instead of once to image and once to projection
    #     self.image_array_fft *= self.whitening_filter * self.whitening_filter
    #     self.image_array = np.fft.ifftshift(self.image_array_fft)
    #     self.image_array = np.fft.ifft2(self.image_array)

    #     self._is_whitened = True

    # def bandpass_filter(
    #     self,
    #     low_resolution_cutoff: float = None,
    #     high_resolution_cutoff: float = None,
    #     low_resolution_decay_rate: float = None,
    #     high_resolution_decay_rate: float = None,
    # ) -> None:
    #     """Apply a bandpass filter to the image (in Fourier space) and recompute the
    #     real-space image. All updates are handled internally meaning the method returns
    #     None.
    #     """
    #     bandpass_filter = get_bandpass_filter(
    #         self.image_array_fft.shape,
    #         self.pixel_size,
    #         low_resolution_cutoff,
    #         high_resolution_cutoff,
    #         low_resolution_decay_rate,
    #         high_resolution_decay_rate,
    #     )

    #     self.image_array_fft *= bandpass_filter
    #     self.image_array = np.fft.ifftshift(self.image_array_fft)
    #     self.image_array = np.fft.ifft2(self.image_array)

    def _validate_to_particle_stack_inputs(
        particle_positions: np.ndarray,
        particle_orientations: np.ndarray = None,
        particle_defocus_parameters: np.ndarray = None,
        particle_z_scores: np.ndarray = None,
        particle_mip_values: np.ndarray = None,
    ) -> None:
        """Helper function to validate inputs for particle extraction.

        Checks that the number of particles is consistent across all provided arrays and
        that the shapes of the arrays match what is expected.

        Args:
            particle_positions: Array of particle positions, shape (N,2)
            particle_orientations: Optional array of orientations, shape (N,3)
            particle_defocus_parameters: Optional array of CTF params, shape (N,3)
            particle_z_scores: Optional array of z-scores, shape (N,)
            particle_mip_values: Optional array of MIP values, shape (N,)
        """
        # Same length validation
        if particle_orientations is not None:
            assert (
                particle_positions.shape[0] == particle_orientations.shape[0]
            ), "Number of particle positions and orientations must be equal."

        if particle_defocus_parameters is not None:
            assert (
                particle_positions.shape[0] == particle_defocus_parameters.shape[0]
            ), "Number of particle positions and CTF parameters must be equal."

        if particle_z_scores is not None:
            assert (
                particle_positions.shape[0] == particle_z_scores.shape[0]
            ), "Number of particle positions and z-scores must be equal."

        if particle_mip_values is not None:
            assert (
                particle_positions.shape[0] == particle_mip_values.shape[0]
            ), "Number of particle positions and MIP values must be equal."

        # Shape validation
        if particle_orientations is not None:
            assert (
                particle_orientations.shape[1] == 3
            ), "Orientation array must have 3 columns."

        if particle_defocus_parameters is not None:
            assert (
                particle_defocus_parameters.shape[1] == 3
            ), "Defocus parameter array must have 3 columns."

        if particle_z_scores is not None:
            assert particle_z_scores.ndim == 1, "Z-score array must be 1-dimensional."

        if particle_mip_values is not None:
            assert (
                particle_mip_values.ndim == 1
            ), "MIP value array must be 1-dimensional."

    def to_particle_stack(
        self,
        box_size: Tuple[int, int],
        particle_positions: np.ndarray,  # In units of pixels  (x, y)
        particle_orientations: np.ndarray = None,  # In radians   (phi, theta, psi)
        particle_defocus_parameters: np.ndarray = None,  # (z1, z2, angle)
        particle_z_scores: np.ndarray = None,
        particle_mip_values: np.ndarray = None,
        position_reference: Literal["center", "corner"] = "center",
    ) -> ParticleStack:
        """Extract particles from the micrograph using the provided particle positions
        and other optional information about each particle.

        Args:
        -----

            box_size (Tuple[int, int]): The size of the box to extract around each
                particle in pixels.
            particle_positions (np.ndarray): The positions of the particles in the
                micrograph in units of pixels. Shape is (N, 2) where N is the number of
                particles. Optional with default of None.
            particle_orientations (np.ndarray): The orientations of the particles in
                radians. Shape is (N, 3) where N is the number of particles. Optional with
                default of None.
            particle_defocus_parameters (np.ndarray): The defocus parameters of the
                particles in Angstroms. Shape is (N, 3) where N is the number of
                particles. Optional with default of None.
            particle_z_scores (np.ndarray): The z-scores of the particles. Shape is (N,)
                where N is the number of particles. Optional with default of None.
            particle_mip_values (np.ndarray): The MIP values of the particles. Shape is
                (N,) where N is the number of particles. Optional with default of None.
            position_reference (Literal["center", "corner"]): Whether the particle
                positions are relative to the center or corner of the bounding box.
                Default is "center".

        Returns:
        --------

            ParticleStack: A ParticleStack object containing the extracted particles.

        """
        _validate_to_particle_stack_inputs(
            particle_positions,
            particle_orientations,
            particle_defocus_parameters,
            particle_z_scores,
            particle_mip_values,
        )

        # Adjust positions if reference point is centered
        if position_reference == "center":
            particle_positions = particle_positions - np.array(box_size) // 2

        particle_images = np.zeros((len(particle_positions), *box_size))
        for i, pos in enumerate(particle_positions):
            x, y = pos
            x, y = int(x), int(y)

            # NOTE: might need to rot 90 or flip for cisTEM vs other conventions
            particle_images[i] = self.image_array[
                y : y + box_size[0], x : x + box_size[1]
            ]

        return ParticleStack(
            pixel_size=self.pixel_size,
            particle_images_array=particle_images,
            orientations=particle_orientations,
            particle_positions=particle_positions,
            particle_defocus_parameters=particle_defocus_parameters,
        )

    def to_json(self) -> dict:
        """Convert the Micrograph object to a JSON-serializable dictionary."""
        return {
            "pixel_size": self.pixel_size,
            "image_path": self.image_path,
            "ctf": self.ctf.to_json() if self.ctf is not None else None,
        }

    @classmethod
    def from_json(cls, json_dict: dict) -> "Micrograph":
        """Create a Micrograph object from a JSON dictionary."""
        # Load the image from the stored path
        with mrcfile.open(json_dict["image_path"]) as mrc:
            image_array = mrc.data.copy()
        
        micrograph = cls(
            image_array=image_array,
            pixel_size=json_dict["pixel_size"],
            image_path=json_dict["image_path"]
        )
        
        if json_dict["ctf"] is not None:
            micrograph.ctf = ContrastTransferFunction.from_json(json_dict["ctf"])
        
        return micrograph
