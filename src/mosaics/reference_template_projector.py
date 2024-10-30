import mrcfile
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import RegularGridInterpolator

from .utils import histogram_2d_gaussian_interpolation
from .utils import calculate_scattering_potential_2d

from abc import ABC, abstractmethod
from typing import Tuple


class AbstractProjector(ABC):
    """Abstract class for taking a 3D template of a structure and projecting it at
    a given orientation. Implemented child classes all have a pixel size attribute (in
    Angstroms) and a shape attribute for the projection, in height x width. Different
    implementations can take in templates in different formats (e.g. 3D electron
    scattering potentials or 3D atomic coordinates)


    """

    pixel_size: float  # in Angstroms
    projection_shape: Tuple[int, int]  # (height, width)

    def __init__(self, pixel_size: float, projection_shape: Tuple[int, int]):
        self.pixel_size = pixel_size
        self.projection_shape = projection_shape

    @abstractmethod
    def get_real_space_projection(
        self, phi: float, theta: float, psi: float
    ) -> np.ndarray:
        """Project the held template at a given orientation, defined by the Euler angles
        in ZYZ convention. Euler angles are in radians.
        """
        raise NotImplementedError("Must be implemented in a subclass")


class DirectCoordinateProjector(AbstractProjector):
    """AbstractProjector implementation that takes in 3D atomic coordinates and projects
    them in real-space (no Fourier slicing). Coordinates are provided in Angstroms and
    can be automatically centered by mass. *NOTE: Current mass centering assumes equal
    atomic weights; this will change in the future.* Atom-wise B-factors are used to
    apply a Gaussian blur to the projections to account for fluctuations in the model.
    These B-factors are in units Angstroms^2.

    Attributes:
        atomic_coordinates (np.ndarray): The 3D atomic coordinates.
        atomic_identities (np.ndarray): The atomic numbers of the atoms.
        b_factors (np.ndarray): The B-factors of the atoms in Angstroms^2.
        
    Methods:
        get_real_space_projection(phi: float, theta: float, psi: float) -> np.ndarray:
            Project the atomic coordinates at a given orientation, defined by the Euler
            angles in ZYZ convention.

    """

    atomic_coordinates: np.ndarray  # Angstroms
    atomic_identities: np.ndarray  # Atomic number
    b_factors: np.ndarray  # Angstroms^2

    def __init__(
        self,
        pixel_size: float,
        projection_shape: Tuple[int, int],
        atomic_coordinates: np.ndarray,
        b_factors: np.ndarray,  # In Angstroms^2
        atomic_identities: np.ndarray,
        center_coords_by_mass: bool = False,
    ):
        super().__init__(pixel_size, projection_shape)
        
        # TODO: Implement validation checks for different inputs
        # TODO: Implement b-factor scaling

        if center_coords_by_mass:
            atomic_coordinates -= np.average(atomic_coordinates, axis=0)
        
        self.atomic_coordinates = atomic_coordinates
        self.atomic_identities = atomic_identities
        self.b_factors = b_factors

    def get_real_space_projection(self, phi, theta, psi) -> np.ndarray:
        """Project the held atomic coordinates at a given orientation, defined by the Euler angles
        in ZYZ convention.

        TODO: finish docstring
        # """
        # # Transform held b-factors into variances for the Gaussian kernel
        # sigma2 = self.b_factors / (8 * np.pi ** 2 * self.pixel_size ** 2)
        # # sigma2 += self.pixel_size ** 2 / 12  # From top-hat on pixel size
        # # sigma2 += 0.27 / self.pixel_size  # Inherent variance fit from PSF?
        
        r = Rotation.from_euler("ZYZ", [phi, theta, psi], degrees=False)

        # Rotate the atomic coordinates, then project onto 2D xy plane
        rotated_coordinates = r.inv().apply(self.atomic_coordinates)
        projected_coordinates = rotated_coordinates[:, :2]

        # Generate bins (in Angstroms) for the histogram centered at (0, 0)
        # NOTE: This might need some tweaking to perfectly match the Fourier slice
        bins0 = np.arange(self.projection_shape[1]) * self.pixel_size
        bins1 = np.arange(self.projection_shape[0]) * self.pixel_size
        bins0 = bins0 - bins0[-1] / 2
        bins1 = bins1 - bins1[-1] / 2

        # projection = histogram_2d_gaussian_interpolation(
        #     x=projected_coordinates[:, 0],
        #     y=projected_coordinates[:, 1],
        #     sigma=np.sqrt(sigma2),  # NOTE: Need to also account for pixel size
        #     bins=(bins0, bins1),
        #     alpha=0.01,
        # )
        
        projection = calculate_scattering_potential_2d(
            x=projected_coordinates[:, 0],
            y=projected_coordinates[:, 1],
            atom_ids=self.atomic_identities,
            b_factors=self.b_factors,
            bins=(bins0, bins1),
            alpha=0.01,
        )

        return projection


class FourierSliceProjector(AbstractProjector):
    """AbstractProjector implementation that takes in a 3D electron scattering potential
    and calculates projections using Fourier slicing. Scattering potentials are assumed
    to be cubic with isotropic pixel size.

    Attributes:
        potential_array (np.ndarray): The 3D scattering potential array.
        potential_array_fft (np.ndarray): The FFT of the scattering potential.
        pixel_size (float): The size of the pixels in Angstroms.

        _interpolator (RegularGridInterpolator): A pre-computed interpolator for the
            spatial frequencies of the scattering potential.

    Methods:
        from_mrc(mrc_path: str) -> ScatteringPotential: Create a ScatteringPotential
            object from an MRC file.
        take_fourier_slice(phi: float, theta: float, psi: float) -> np.ndarray: Takes a
            Fourier slice of the scattering potential at an orientation from the given
            Euler angles.
        take_real_space_projection(phi: float, theta: float, psi: float) -> np.ndarray:
            Take a real-space projection of the scattering potential at an orientation
            from the given Euler angles.
    """

    potential_array: np.ndarray
    potential_array_fft: np.ndarray

    _interpolator: RegularGridInterpolator

    @classmethod
    def from_mrc(cls, mrc_path: str):
        """Create a ScatteringPotential object from an MRC file."""
        with mrcfile.open(mrc_path) as mrc:
            potential_array = mrc.data.copy()
            pixel_size = mrc.voxel_size.x

        # Conform to cisTEM convention (reversed axes)
        potential_array = np.swapaxes(potential_array, 0, -1)

        return cls(potential_array, pixel_size)

    def __init__(self, potential_array: np.ndarray, pixel_size: float):
        projection_shape = (potential_array.shape[0], potential_array.shape[1])

        super().__init__(pixel_size, projection_shape)

        self.potential_array = potential_array

        # Precompute the FFT of the scattering potential
        # NOTE: The real-space potential is first fft-shifted to correct for the
        # odd-valued frequencies when taking a Fourier slice. See (TODO) for more info
        self.potential_array_fft = np.fft.fftshift(potential_array)
        self.potential_array_fft = np.fft.fftn(self.potential_array_fft)
        self.potential_array_fft = np.fft.fftshift(self.potential_array_fft)

        dim = [np.arange(s) - s // 2 for s in potential_array.shape]
        self._interpolator = RegularGridInterpolator(
            points=dim,
            values=self.potential_array_fft,
            method="linear",
            bounds_error=False,
            fill_value=0,
        )

    def take_fourier_slice(self, phi: float, theta: float, psi: float) -> np.ndarray:
        """Takes a Fourier slice of the pre-computed scattering potential at an
        orientation from the given Euler angles. The angles are in radians, and the
        rotation convention is ZYZ.

        Returned array is in the Fourier domain and centered (zero-frequency in center)

        Args:
            phi (float): The rotation around the Z axis in radians.
            theta (float): The rotation around the Y' axis in radians.
            psi (float): The rotation around the Z'' axis in radians.

        Returns:
            np.ndarray: The Fourier slice of the scattering potential.
        """
        rot = Rotation.from_euler("ZYZ", [phi, theta, psi])

        # Generate a grid of integer coordinates at z = 0 then rotate
        x = np.arange(self.projection_shape[0]) - self.projection_shape[0] // 2
        y = np.arange(self.projection_shape[1]) - self.projection_shape[1] // 2
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros_like(xx)

        coordinates = np.stack([xx, yy, zz], axis=-1)
        coordinates = coordinates.reshape(-1, 3)
        coordinates = rot.apply(coordinates)

        # Interpolate the scattering potential at the rotated coordinates
        fourier_slice = self._interpolator(coordinates)
        fourier_slice = fourier_slice.reshape(xx.shape)

        return fourier_slice

    def get_real_space_projection(
        self, phi: float, theta: float, psi: float
    ) -> np.ndarray:
        """Take a real-space projection of the scattering potential at an orientation
        from the given Euler angles. The angles are in radians, and the rotation
        convention is ZYZ.

        Returned array is in real-space.

        Args:
            phi (float): The rotation around the Z axis in radians.
            theta (float): The rotation around the Y' axis in radians.
            psi (float): The rotation around the Z'' axis in radians.

        Returns:
            np.ndarray: The projection of the scattering potential.
        """
        fourier_slice = self.take_fourier_slice(phi, theta, psi)

        fourier_slice = np.fft.ifftshift(fourier_slice)
        projection = np.fft.ifftn(fourier_slice)
        projection = np.fft.ifftshift(projection)

        return projection
