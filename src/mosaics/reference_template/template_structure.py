from typing import Optional
from typing import Union

import mmdf
import numpy as np
import torch
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
    atomwise_potentials: np.ndarray  # (n, h, w, d) array of potentials
    atomwise_voxel_positions: np.ndarray
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

    def get_masked_volume(
        self, mask_indices: np.ndarray, invert: bool = False
    ) -> np.ndarray:
        """Return the real=space volume with the indicated atoms masked.

        Args:
            mask_indices (np.ndarray): The indices of the atoms remove.
            invert (bool): If invert is True, then only the atoms at the
                indices are kept.

        Returns:
            np.ndarray: The masked volume.
        """
        # Remove the mask_indices from all idxs if invert is False
        if not invert:
            mask_indices = np.setdiff1d(
                np.arange(self.atom.size), mask_indices
            )

        # Calculate the upsampled volume from the atomwise potentials
        final_volume = torch.zeros(
            self.upsampled_volume_shape, dtype=torch.float32
        )
        final_volume = place_voxel_neighborhoods_in_volume(
            neighborhood_potentials=self.atomwise_potentials[mask_indices],
            voxel_positions=self.atomwise_voxel_positions[mask_indices],
            final_volume=final_volume,
        )

        # final_volume = torch.fft.fftshift(final_volume, dim=(-3, -2, -1))
        # final_volume_FFT = torch.fft.rfftn(final_volume, dim=(-3, -2, -1))

        # Apply the necessary filtering to the volume
        # NOTE: All these filters are independent of the mask. Should
        # pre-calculate these as well since their application takes a
        # non-trivial amount of time.
        # BUT not for this class to hold reference to.
        # TODO

        return final_volume
