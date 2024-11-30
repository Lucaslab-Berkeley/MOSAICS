import mrcfile
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch_fourier_slice import extract_central_slices_rfft_3d


class FourierSliceProjector:
    """Takes in a 3D electron scattering potential and calculates projections
    using Fourier slicing. Scattering potentials are assumed to be cubic with
    isotropic pixel size.

    Attributes:
        potential_array (torch.Tensor): The 3D scattering potential array.
        potential_array_rfft (torch.Tensor): The FFT of the scattering
            potential.
        pixel_size (float): The size of the pixels in Angstroms.

        # _interpolator (RegularGridInterpolator): A pre-computed interpolator
        #     for the spatial frequencies of the scattering potential.

    Methods:
        from_mrc(mrc_path: str) -> ScatteringPotential: Create a
            ScatteringPotential object from an MRC file.
        take_fourier_slice(
            phi: torch.Tensor, theta: torch.Tensor, psi: torch.Tensor
        ) -> torch.Tensor:
            Takes a Fourier slice of the held scattering potential RFFT at the
            given orientation. The angles are in radians, and the rotation
            convention is ZYZ.
        get_real_space_projection(
            phi: torch.Tensor, theta: torch.Tensor, psi: torch.Tensor
        ) -> torch.Tensor:
            Constructs a real-space projection of the scattering potential at
            the given orientation. The angles are in radians, and the rotation
            convention is ZYZ. Returned tensor is in real space
    """

    potential: torch.Tensor
    potential_rfft: torch.Tensor

    @classmethod
    def from_mrc(cls, mrc_path: str):
        """Create a ScatteringPotential object from an MRC file."""
        with mrcfile.open(mrc_path) as mrc:
            potential_array = mrc.data.copy()
            pixel_size = mrc.voxel_size.x

        # Conform to cisTEM convention (reversed axes)
        # TODO: Check the header information for axis order rather than assume
        potential_array = np.swapaxes(potential_array, 0, -1)
        potential = torch.tensor(potential_array, dtype=torch.float32)

        return cls(potential, pixel_size)

    def __init__(
        self,
        potential: torch.Tensor,
        pixel_size: float,
        potential_rfft: torch.Tensor = None,
    ):
        assert isinstance(
            potential, torch.Tensor
        ), "potential must be a torch.Tensor"

        self.pixel_size = pixel_size
        self.potential = potential

        # Calculate the RFFT if not provided
        if potential_rfft is None:
            # potential_rfft = torch.fft.fftshift(potential)
            potential_rfft = torch.fft.fftn(potential)
            potential_rfft = torch.fft.fftshift(potential_rfft)

        self.potential_rfft = potential_rfft

    def take_fourier_slice(
        self, psi: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor
    ) -> np.ndarray:
        """Takes a Fourier slice of the pre-computed scattering potential at an
        orientation from the given Euler angles. The angles are in radians, and
        the rotation convention is ZYZ.

        Returned array is in the Fourier domain and centered (zero-frequency in
        center)

        Args:
            psi (torch.Tensor): The rotation around the Z'' axis in radians.
            theta (torch.Tensor): The rotation around the Y' axis in radians.
            phi (torch.Tensor): The rotation around the Z axis in radians.

        Returns:
            np.ndarray: The Fourier slice of the scattering potential.
        """
        # TODO: Keep all computation in PyTorch
        psi_np = psi.cpu().numpy()
        theta_np = theta.cpu().numpy()
        phi_np = phi.cpu().numpy()
        angles = np.stack([phi_np, theta_np, psi_np], axis=-1)

        rot_np = Rotation.from_euler("ZYZ", angles, degrees=False)
        rot = torch.Tensor(rot_np.as_matrix())

        # Use torch_fourier_slice to take the Fourier slice
        fourier_slice = extract_central_slices_rfft_3d(
            volume_rfft=self.potential_rfft,
            image_shape=self.potential.shape,
            rotation_matrices=rot,
        )

        return fourier_slice

    def get_real_space_projection(
        self, psi: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor
    ) -> torch.Tensor:
        """Take a real-space projection of the scattering potential at an
        orientation from the given Euler angles. The angles are in radians, and
        the rotation convention is ZYZ.

        Returned array is in real-space.

        Args:
            phi (torch.Tensor): The rotation around the Z axis in radians.
            theta (torch.Tensor): The rotation around the Y' axis in radians.
            psi (torch.Tensor): The rotation around the Z'' axis in radians.

        Returns:
            torch.Tensor: The projection of the scattering potential.
        """
        fourier_slice = self.take_fourier_slice(phi, theta, psi)

        # fourier_slice = torch.fft.ifftshift(fourier_slice, dim=(-2, -1))
        projection = torch.fft.irfftn(fourier_slice, dim=(-2, -1))
        # projection = torch.fft.ifftshift(projection, dim=(-2, -1))

        return projection
