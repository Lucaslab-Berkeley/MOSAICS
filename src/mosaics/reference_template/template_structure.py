from typing import Optional
from typing import Union

import mmdf
import numpy as np
import torch

# from torch_fourier_slice.dft_utils import ifftshift_2d
# from torch_fourier_slice.dft_utils import fftshift_2d
from torch_fourier_slice.dft_utils import fftshift_3d
from torch_fourier_slice.dft_utils import ifftshift_3d
from ttsim3d.grid_coords import fourier_rescale_3d_force_size
from ttsim3d.simulate3d import _calculate_lead_term
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
]
RNA_RESIDUES = ["A", "C", "G", "U"]


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

    def set_chain_order(self, chain_order: np.ndarray) -> None:
        """Set the order of the chains in the structure.

        Args:
            chain_order (np.ndarray): The order of the chains.
        """
        if set(chain_order) != set(self.chain):
            raise ValueError(
                "Members of `chain_order` do not match the held chains. "
                f"Missing: {set(self.chain) - set(chain_order)}"
                f"Extra:   {set(chain_order) - set(self.chain)}"
            )

        # Get the current ordering of the chains. NOTE: This assumes the
        # chains are already ordered (e.g. [AAABBCCCC] not [ABBACAC])
        current_chain_index = {
            chain: np.where(self.chain == chain)[0]
            for chain in np.unique(self.chain)
        }
        new_index = np.concatenate(
            [current_chain_index[chain] for chain in chain_order]
        )

        # Re-index all of the attribute arrays
        self.model = self.model[new_index]
        self.chain = self.chain[new_index]
        self.residue = self.residue[new_index]
        self.residue_id = self.residue_id[new_index]
        self.atom = self.atom[new_index]
        self.element = self.element[new_index]
        self.atomic_number = self.atomic_number[new_index]
        self.atomic_weight = self.atomic_weight[new_index]
        self.x = self.x[new_index]
        self.y = self.y[new_index]
        self.z = self.z[new_index]
        self.b_factor = self.b_factor[new_index]

        self.atomwise_potentials = self.atomwise_potentials[new_index]

    def randomize_chain_order(self) -> None:
        """Randomize the order of the chains in the structure."""
        new_chain_order = np.random.permutation(np.unique(self.chain))

        self.set_chain_order(new_chain_order)

    def _remove_atoms_by_index(
        self, atom_idxs: np.ndarray
    ) -> "TemplateStructure":
        """Remove atoms from the structure by their index.

        Args:
            atom_idxs (np.ndarray): The indices of the atoms to remove.

        Returns:
            TemplateStructure: A new structure with the atoms removed.
        """
        # Invert the indices to keep the atoms
        atom_idxs = np.setdiff1d(np.arange(self.atom.size), atom_idxs)

        new_template_structure = TemplateStructure(
            model=self.model[atom_idxs],
            chain=self.chain[atom_idxs],
            residue=self.residue[atom_idxs],
            residue_id=self.residue_id[atom_idxs],
            atom=self.atom[atom_idxs],
            element=self.element[atom_idxs],
            atomic_number=self.atomic_number[atom_idxs],
            atomic_weight=self.atomic_weight[atom_idxs],
            x=self.x[atom_idxs],
            y=self.y[atom_idxs],
            z=self.z[atom_idxs],
            b_factor=self.b_factor[atom_idxs],
            pixel_size=self.pixel_size,
            volume_shape=self.volume_shape,
        )

        return new_template_structure

    def remove_atoms_from_residues(
        self,
        chains: Union[list[str], np.ndarray],
        residue_ids: Union[list[int], np.ndarray],
        inplace: bool = False,
        return_removed_atoms: bool = False,
    ) -> Optional["TemplateStructure"]:
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
        inplace (bool): Optional boolean specifying Whether to modify the
            structure object in place. Default is False a new structure is
            returned.
        return_removed_atoms (bool): Optional boolean specifying whether to
            return the removed atoms as a new TemplateStructure object. Default
            is False.

        Returns:
        --------


        """
        removed_atom_idxs = []  # These are per-atom indices

        # Iterate over each chain and residue ID to remove atoms from
        for chain, residue_id in zip(chains, residue_ids):
            idxs = np.where(
                (self.chain == chain) & (self.residue_id == residue_id)
            )[0]

            print(idxs)

            # Get the residue type
            _is_amino_acid = self.residue[idxs[0]] in AMINO_ACID_RESIDUES
            _is_rna = self.residue[idxs[0]] in RNA_RESIDUES

            # Which atoms to remove depends on the residue type
            if _is_amino_acid:
                atom_names = ["N", "CA", "C", "O"]
            elif _is_rna:
                atom_names = ["C1'", "C2'", "C3'", "C4'", "O4'"]

            for i in idxs:
                if self.atom[i] in atom_names:
                    removed_atom_idxs.append(i)

        # Invert the indices if we want to keep the removed atoms
        if return_removed_atoms:
            removed_atom_idxs = np.setdiff1d(
                np.arange(self.atom.size), removed_atom_idxs
            )

        # Choose inplace or return a new structure
        if not inplace:
            return self._remove_atoms_by_index(removed_atom_idxs)
        else:
            self = self._remove_atoms_by_index(removed_atom_idxs)

        return None

    def _calculate_atomwise_potentials(self):
        """Helper function for calculating scatterin
        g potentials from model

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
            atom_b_factors=atom_b_factors,
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

        # Early return if no upsampling is requested
        # TODO: Better code organization for this method
        if (
            self.upsampling == 1
            or self.upsampled_pixel_size == self.pixel_size
        ):
            return volume_upsampled

        volume_upsampled = fftshift_3d(volume_upsampled, rfft=False)
        volume_upsampled_RFFT = torch.fft.rfftn(
            volume_upsampled, dim=(-3, -2, -1)
        )

        # TODO: Exposure filtering on upsampled volume

        volume_RFFT = fourier_rescale_3d_force_size(
            volume_fft=volume_upsampled_RFFT,
            volume_shape=self.upsampled_volume_shape,
            target_size=self.volume_shape[0],  # TODO: pass this arg as a tuple
            rfft=True,
            fftshift=False,
        )

        # TODO: MTF filtering on volume_RFFT

        # Cropping and inverse RFFT
        final_volume = torch.fft.irfftn(
            volume_RFFT,
            s=(
                self.volume_shape[0],
                self.volume_shape[0],
                self.volume_shape[0],
            ),
            dim=(-3, -2, -1),
        )
        final_volume = ifftshift_3d(final_volume, rfft=False)

        return final_volume
