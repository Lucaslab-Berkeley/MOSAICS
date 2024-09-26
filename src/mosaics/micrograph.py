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
    image_array_fft: np.ndarray
    # power_spectral_density: np.ndarray = None
    whitening_filter: np.ndarray = None  # for holding pre-computation
    pixel_size: float  # In Angstroms, assume square pixels

    _is_whitened: bool = False

    def from_mrc(cls, mrc_path: str):
        """Create a Micrograph object from an MRC file."""
        with mrcfile.open(mrc_path) as mrc:
            image_array = mrc.data.copy()
            pixel_size = mrc.voxel_size.x

        return cls(image_array, pixel_size)

    def __init__(self, image_array: np.ndarray, pixel_size: float):
        self.image_array = image_array
        self.pixel_size = pixel_size

        # Pre-compute the FFT of the image
        self.image_array_fft = np.fft.fft2(image_array)
        self.image_array_fft = np.fft.fftshift(self.image_array_fft)

    def whiten_image(self) -> None:
        """Apply a whitening filter to the image (in Fourier space) and recompute the
        real-space image. All updates are handled internally meaning the method returns
        None.
        """
        if self._is_whitened:
            return

        self.whitening_filter = get_whitening_filter(self.image_array, self.pixel_size)

        # Re-compute the image
        # filter applied twice to image instead of once to image and once to projection
        self.image_array_fft *= self.whitening_filter * self.whitening_filter
        self.image_array = np.fft.ifftshift(self.image_array_fft)
        self.image_array = np.fft.ifft2(self.image_array)

        self._is_whitened = True

    def bandpass_filter(
        self,
        low_resolution_cutoff: float = None,
        high_resolution_cutoff: float = None,
        low_resolution_decay_rate: float = None,
        high_resolution_decay_rate: float = None,
    ) -> None:
        """Apply a bandpass filter to the image (in Fourier space) and recompute the
        real-space image. All updates are handled internally meaning the method returns
        None.
        """
        bandpass_filter = get_bandpass_filter(
            self.image_array_fft.shape,
            self.pixel_size,
            low_resolution_cutoff,
            high_resolution_cutoff,
            low_resolution_decay_rate,
            high_resolution_decay_rate,
        )

        self.image_array_fft *= bandpass_filter
        self.image_array = np.fft.ifftshift(self.image_array_fft)
        self.image_array = np.fft.ifft2(self.image_array)

    def to_particle_stack(
        self,
        box_size: Tuple[int, int],
        particle_positions: np.ndarray,  # In units of pixels
        particle_orientations: np.ndarray,  # In radians
        particle_defocus_parameters: np.ndarray = None,
        position_reference: Literal["center", "corner"] = "center",
    ) -> ParticleStack:
        """Extract particles from the micrograph and return a ParticleStack object.

        TODO: complete docstring

        TODO: complete method
        """
        # Do input validation
        assert (
            particle_positions.shape[0] == particle_orientations.shape[0]
        ), "Number of particle positions and orientations must be equal."
        if particle_defocus_parameters is not None:
            assert (
                particle_positions.shape[0] == particle_defocus_parameters.shape[0]
            ), "Number of particle positions and CTF parameters must be equal."

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

        raise NotImplementedError
