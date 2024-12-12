from typing import Iterator
from typing import Literal
from typing import Optional
from typing import Union

import mmdf
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch_fourier_filter.ctf import calculate_ctf_2d
from torch_fourier_slice import extract_central_slices_rfft_3d
from torch_fourier_slice.dft_utils import fftshift_2d
from torch_fourier_slice.dft_utils import fftshift_3d
from torch_fourier_slice.dft_utils import ifftshift_2d
from ttsim3d.simulate3d import _calculate_lead_term
from ttsim3d.simulate3d import apply_simulation_filters
from ttsim3d.simulate3d import place_voxel_neighborhoods_in_volume
from ttsim3d.simulate3d import simulate_atomwise_scattering_potentials

AMINO_ACID_RESIDUES = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",  # Unknown residue
]
RNA_RESIDUES = ["A", "C", "G", "U"]


def sliding_window_iterator(
    length: int, width: int, increment: int
) -> Iterator[np.ndarray]:
    """Generator function 1-dimensional sliding window indexes.

    Each iteration yields a list/array of indexes with length `width`. The
    first index is `0` and subsequent indexes are incremented by `increment`
    until the end of the array is reached. Note that the last window may be
    smaller than `width` if there is not perfect factorization.

    Args:
        length (int): The length of the array to slide over.
        width (int): The width of the sliding window.
        increment (int): The increment of the sliding window.

    Yields:
        np.ndarray: The indexes of the sliding window at each iteration.

    Example:
        >>> for idx in sliding_window_iterator(10, 3, 2):
        ...     print(idx)
        [0 1 2]
        [2 3 4]
        [4 5 6]
        [6 7 8]
        [8, 9]
    """
    for i in range(0, length - width + increment, increment):
        yield np.arange(i, min(i + width, length))


class TemplateStructure:
    """Holds reference to a 3D structure parsed from a .pdb or .cif file.

    TODO: finish docstring
    """

    # Attributes associated with an atomic structure
    model: np.ndarray
    chain: np.ndarray
    residue: np.ndarray
    residue_id: np.ndarray
    atom: np.ndarray
    element: np.ndarray
    atomic_number: np.ndarray
    atomic_weight: np.ndarray
    # covalent_radius: np.ndarray
    # van_der_waals_radius: np.ndarray
    # heteroatom_flag: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    # charge: np.ndarray
    # occupancy: np.ndarray
    b_factor: np.ndarray

    # Attributes associated with a 3D volume for Fourier slicing
    atomwise_potentials: np.ndarray  # (n, h*w*d) array of potentials
    atomwise_voxel_positions: np.ndarray  # (n, h*w*d, 3) array of integers
    pixel_size: float
    volume_shape: tuple

    upsampling: int
    upsampled_pixel_size: float
    upsampled_volume_shape: tuple

    @classmethod
    def from_file(
        cls,
        file_path: str,
        pixel_size: float,
        volume_shape: tuple[int, int, int],
        upsampling: int = -1,
        center_by_mass: bool = True,
    ) -> "TemplateStructure":
        """Create a TemplateStructure object from a .pdb or .cif file.

        Args:
            file_path (str): The path to the .pdb or .cif file.

        Returns:
            TemplateStructure: The parsed structure.
        """
        df = mmdf.read(file_path)

        return cls(
            model=df["model"].values,
            chain=df["chain"].values,
            residue=df["residue"].values,
            residue_id=df["residue_id"].values,
            atom=df["atom"].values,
            element=df["element"].values,
            atomic_number=df["atomic_number"].values,
            atomic_weight=df["atomic_weight"].values,
            x=df["x"].values,
            y=df["y"].values,
            z=df["z"].values,
            b_factor=df["b_isotropic"].values,
            pixel_size=pixel_size,
            volume_shape=volume_shape,
            upsampling=upsampling,
            center_by_mass=center_by_mass,
        )

    def __init__(
        self,
        model: np.ndarray,
        chain: np.ndarray,
        residue: np.ndarray,
        residue_id: np.ndarray,
        atom: np.ndarray,
        element: np.ndarray,
        atomic_number: np.ndarray,
        atomic_weight: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        b_factor: np.ndarray,
        pixel_size: float,
        volume_shape: tuple,
        upsampling: int = -1,
        center_by_mass: bool = True,
    ):
        self.model = model
        self.chain = chain
        self.residue = residue
        self.residue_id = residue_id
        self.atom = atom
        self.element = element
        self.atomic_number = atomic_number
        self.atomic_weight = atomic_weight
        self.x = x
        self.y = y
        self.z = z
        self.b_factor = b_factor

        self.pixel_size = pixel_size
        self.volume_shape = volume_shape
        self.upsampling = upsampling

        # Center by mass, if requested, before potential calculation
        if center_by_mass:
            self.center_by_mass()

        # Calculate the atomwise potentials
        self._calculate_atomwise_potentials()

    def center_by_mass(self) -> None:
        """Transform the structure so center of mass is at (0, 0, 0)."""
        center_of_mass = np.array([self.x, self.y, self.z])
        center_of_mass = np.dot(center_of_mass, self.atomic_weight)
        center_of_mass /= np.sum(self.atomic_weight)

        self.x -= center_of_mass[0]
        self.y -= center_of_mass[1]
        self.z -= center_of_mass[2]

    # def rotate(self, phi: float, theta: float, psi: float) -> None:
    #     """Rotate the structure by the given Euler angles."""
    #     rotation_matrix = euler2rot(phi, theta, psi)

    #     coordinates = np.array([self.x, self.y, self.z]).T
    #     coordinates = np.dot(coordinates, rotation_matrix)

    #     self.x = coordinates[:, 0]
    #     self.y = coordinates[:, 1]
    #     self.z = coordinates[:, 2]

    def chain_residue_pairs(
        self,
        chain_order: Optional[str | list[str]] = None,
        residue_types: Literal["both", "amino_acid", "rna"] = "both",
    ) -> list[tuple[str, int]]:
        """Generate chain-residue pairs based on the specified residue types.

        Parameters:
        -----------
        chain_order (str): The ordering of the chains. If None, the chains are
            in sequential order. Otherwise should be a string where characters
            are the chain identifiers in the requested order.
        residue_types (Literal["both", "amino_acid", "rna"]): The residue types
            to include in the pairs. Default is "both".

        Returns:
        --------
        list[tuple[str, int]]: The chain-residue pairs.
        """
        assert residue_types in ["both", "amino_acid", "rna"]

        # Indexes of which residues to include
        if residue_types == "both":
            keep_idxs = np.ones_like(self.residue, dtype=bool)
        elif residue_types == "amino_acid":
            keep_idxs = np.isin(self.residue, AMINO_ACID_RESIDUES)
        elif residue_types == "rna":
            keep_idxs = np.isin(self.residue, RNA_RESIDUES)

        # dictionary has keys of unique chains and values of sorted residue IDs
        chain_residue_id_dict = {
            c: np.sort(np.unique(self.residue_id[self.chain == c]))
            for c in set(self.chain[keep_idxs])
        }

        # Check that the requested chain order is valid (or populate)
        if chain_order is None:
            chain_order = sorted(set(chain_residue_id_dict.keys()))
        else:
            assert len(set(chain_order)) == len(
                chain_order
            ), "The chain order must not contain duplicate chain identifiers."
            assert set(chain_order) == set(chain_residue_id_dict.keys()), (
                "The chain order must contain all chain identifiers.\n"
                "Expected to contain: "
                f"{sorted(set(chain_residue_id_dict.keys()))}.\n"
                f"Received (sorted)  : {sorted(set(chain_order))}."
            )

        # Finally, create the pairs
        pairs = []
        for chain in chain_order:
            for residue_id in chain_residue_id_dict[chain]:
                pairs.append((chain, residue_id))

        return pairs

    def get_removed_atoms_indexes(
        self,
        chains: Union[list[str], np.ndarray],
        residue_ids: Union[list[int], np.ndarray],
        return_removed_atoms: bool = False,
    ) -> np.ndarray:
        """Removes certain atoms from residues in the structure; these residues
        are indexed by their chain and residue ID. Atoms removed depend on the
        residue type with amino acids removing (N, CA, C, O) and nucleic acids
        removing (C1', C2', C3', C4', O4').

        Args:
        -----

        chains (Union[list[str], np.ndarray]): The chain identifiers for
            residues to remove atoms from.
        residue_ids (Union[list[int], np.ndarray]): The residue IDs to remove
            atoms from. These ids correspond to the residue number in the the
            chain.
        return_removed_atoms (bool): Optional boolean specifying whether to
            return the removed atoms as a new TemplateStructure object. Default
            is False.

        Returns:
        --------


        """
        # Create masks for amino acid and RNA atoms
        AMINO_ACID_ATOMS = ["N", "CA", "C", "O"]
        RNA_ATOMS = ["C1'", "C2'", "C3'", "C4'", "O4'"]

        # Attempt to cast residue_ids as integers
        residue_ids = np.array(residue_ids, dtype=int)
        chains = np.array(chains)

        # Loop through each pair to find indexes to target
        idxs = []
        for c, r in zip(chains, residue_ids):
            # Find indices where both chain and residue_id match
            idx = np.where((self.chain == c) & (self.residue_id == r))[0]
            if len(idx) != 0:
                idxs.extend(idx)

        idxs = np.array(idxs)

        # Determine the residue type for each filtered index
        _is_aa = np.isin(self.residue[idxs], AMINO_ACID_RESIDUES)
        _is_rna = np.isin(self.residue[idxs], RNA_RESIDUES)
        _is_removed_aa_atom = np.isin(self.atom[idxs], AMINO_ACID_ATOMS)
        _is_removed_rna_atom = np.isin(self.atom[idxs], RNA_ATOMS)

        removed_atom_idxs = idxs[
            (_is_aa & _is_removed_aa_atom) | (_is_rna & _is_removed_rna_atom)
        ]

        # Invert the indices, if necessary to return FL - mask
        if not return_removed_atoms:
            removed_atom_idxs = np.setdiff1d(
                np.arange(self.atom.size), removed_atom_idxs
            )

        return removed_atom_idxs

    def _calculate_atomwise_potentials(self):
        """Helper function for calculating scattering potentials from model

        TODO: finish docstring
        """
        atom_pos_zyx = np.array([self.z, self.y, self.x]).T
        atom_ids = [element.upper() for element in self.element]
        atom_b_factors = self.b_factor
        sim_pixel_spacing = self.pixel_size
        sim_volume_shape = self.volume_shape

        # NOTE: 300.0 is hardcoded and needs to be grabbed from another class
        lead_term = _calculate_lead_term(300.0, sim_pixel_spacing)

        # Cast everything to torch tensors
        # TODO: Refactor codebase to use torch tensors
        atom_pos_zyx = torch.tensor(atom_pos_zyx, dtype=torch.float32)
        atom_b_factors = torch.tensor(atom_b_factors, dtype=torch.float32)

        scattering_results = simulate_atomwise_scattering_potentials(
            atom_positions_zyx=atom_pos_zyx,
            atom_ids=atom_ids,
            # atom_b_factors=atom_b_factors,
            atom_b_factors=torch.ones_like(atom_b_factors) * 60.0 * 0.25,
            sim_pixel_spacing=sim_pixel_spacing,
            sim_volume_shape=sim_volume_shape,
            lead_term=lead_term,
            upsampling=self.upsampling,
        )

        self.atomwise_potentials = scattering_results[
            "neighborhood_potentials"
        ]
        self.atomwise_voxel_positions = scattering_results["voxel_positions"]
        self.upsampling = scattering_results["upsampling"]
        self.upsampled_pixel_size = scattering_results["upsampled_pixel_size"]
        self.upsampled_volume_shape = scattering_results["upsampled_shape"]

    def volume_from_atoms(self, atom_idxs: torch.Tensor = None) -> np.ndarray:
        """Return the real=space volume with the indicated atoms masked.

        Args:
            atom_idxs (np.ndarray): The indices of the atoms to place into
            the volume. If None, then all atoms are placed.

        Returns:
            np.ndarray: The volume.
        """
        # Remove the mask_indices from all idxs if invert is False
        if atom_idxs is None:
            atom_idxs = torch.arange(self.atomwise_potentials.size(0))

        volume_upsampled = torch.zeros(
            self.upsampled_volume_shape, dtype=torch.float32
        )
        volume_upsampled = place_voxel_neighborhoods_in_volume(
            neighborhood_potentials=self.atomwise_potentials[atom_idxs],
            voxel_positions=self.atomwise_voxel_positions[atom_idxs],
            final_volume=volume_upsampled,
        )

        # Convert to Fourier space for filtering
        volume_upsampled = torch.fft.fftshift(
            volume_upsampled, dim=(-3, -2, -1)
        )
        upsampled_volume_FFT = torch.fft.rfftn(
            volume_upsampled, dim=(-3, -2, -1)
        )

        mtf_filename = "/home/mgiammar/MOSAICS/src/mosaics/reference_template/mtf_k2_300kV.star"  # noqa: E501
        final_volume = apply_simulation_filters(
            upsampled_volume_rfft=upsampled_volume_FFT,
            upsampled_shape=self.upsampled_volume_shape,
            final_shape=self.volume_shape,
            upsampled_pixel_size=self.upsampled_pixel_size,
            upsampling=self.upsampling,
            dose_weighting=True,  # NOTE: Currently hardcoded
            num_frames=30,  # NOTE: Currently hardcoded
            fluence_per_frame=1.0,  # NOTE: Currently hardcoded
            dose_B=-1,  # NOTE: Currently hardcoded
            modify_signal=2,  # NOTE: Currently hardcoded
            apply_dqe=True,  # NOTE: Currently hardcoded
            mtf_filename=mtf_filename,  # NOTE: Currently hardcoded
        )

        return final_volume


def _get_fourier_slices_rfft(
    volume: torch.Tensor,
    phi: torch.Tensor,
    theta: torch.Tensor,
    psi: torch.Tensor,
    degrees: bool = True,
):
    """Helper function to get Fourier slices of a volume."""
    shape = volume.shape
    volume_rfft = fftshift_3d(volume, rfft=False)
    volume_rfft = torch.fft.fftn(volume_rfft, dim=(-3, -2, -1))
    volume_rfft = fftshift_3d(volume_rfft, rfft=True)

    # TODO: Keep all computation in PyTorch
    psi_np = psi.cpu().numpy()
    theta_np = theta.cpu().numpy()
    phi_np = phi.cpu().numpy()

    angles = np.stack([phi_np, theta_np, psi_np], axis=-1)

    rot_np = Rotation.from_euler("zyz", angles, degrees=degrees)
    rot = torch.Tensor(rot_np.as_matrix())

    # Use torch_fourier_slice to take the Fourier slice
    fourier_slices = extract_central_slices_rfft_3d(
        volume_rfft=volume_rfft,
        image_shape=shape,
        rotation_matrices=rot,
    )

    # Invert contrast to match image
    fourier_slices = -fourier_slices

    return fourier_slices


def _rfft_slices_to_real_projections(
    fourier_slices: torch.Tensor,
) -> torch.Tensor:
    """Convert Fourier slices to real-space projections.

    NOTE: Assumes 2d or batched 2d input

    TODO: docstring
    """
    fourier_slices = ifftshift_2d(fourier_slices, rfft=True)
    projections = torch.fft.irfftn(fourier_slices, dim=(-2, -1))
    projections = ifftshift_2d(projections, rfft=False)

    return projections


def get_real_space_projections(
    volume: torch.Tensor,
    phi: torch.Tensor,
    theta: torch.Tensor,
    psi: torch.Tensor,
    degrees: bool = True,
) -> torch.Tensor:
    """Generate real-space projections of a volume using Fourier slicing.

    Note that rotations are applied using the 'zyz' convention with Euler
    angles phi, theta, and psi. The volume is assumed to be in real space.
    Returned tensor is also in real space.

    Parameters
    ----------
    volume : torch.Tensor
        The volume to project, in real space.
    phi : torch.Tensor
        The phi angles of the projections, in radians.
    theta : torch.Tensor
        The theta angles of the projections, in radians.
    psi : torch.Tensor

    Returns
    -------
    torch.Tensor
        The real-space projections with shape (n_projections, n_pixels,
        n_pixels).

    """
    fourier_slices = _get_fourier_slices_rfft(volume, phi, theta, psi, degrees)

    return _rfft_slices_to_real_projections(fourier_slices)


def get_real_space_projections_ctf(
    volume: torch.Tensor,
    phi: torch.Tensor,
    theta: torch.Tensor,
    psi: torch.Tensor,
    defocus1: torch.Tensor,
    defocus2: torch.Tensor,
    astigmatism_angle: torch.Tensor,
    degrees: bool = True,
) -> torch.Tensor:
    """Construct real-space projections with CTF and whitening applied.

    Parameters
    ----------
    volume : torch.Tensor
        The volume to project, in real space.
    phi : torch.Tensor
        The phi angles of the projections.
    theta : torch.Tensor
        The theta angles of the projections.
    psi : torch.Tensor
        The psi angles of the projections.
    defocus1 : torch.Tensor
        The defocus value for the first CTF filter, in Angstroms.
    defocus2 : torch.Tensor
        The defocus value for the second CTF filter, in Angstroms.
    astigmatism_angle : torch.Tensor
        The angle of the astigmatism, in degrees
    degrees : bool, optional
        Whether the angles are in degrees or radians, by default True.
    """
    fourier_slices = _get_fourier_slices_rfft(volume, phi, theta, psi, degrees)

    # Calculate the CTF filters for each slice
    defocus = (defocus1 + defocus2) / 2
    astigmatism = (defocus1 - defocus2) / 2
    ctf_filters = calculate_ctf_2d(
        defocus=defocus,
        astigmatism=astigmatism,
        astigmatism_angle=astigmatism_angle,
        voltage=300,  # TODO: unhardcode these
        spherical_aberration=2.7,
        amplitude_contrast=0.07,
        b_factor=0,
        phase_shift=0.0,
        pixel_size=1.06,
        image_shape=(volume.shape[0], volume.shape[0]),
        rfft=True,
        fftshift=False,
    )
    ctf_filters = fftshift_2d(ctf_filters, rfft=True)

    # Apply the CTF filters to the Fourier slices
    fourier_slices = fourier_slices * ctf_filters

    return _rfft_slices_to_real_projections(fourier_slices)
