"""Module for different ways of generating alternate templates for comparison."""

from abc import abstractmethod
from collections.abc import Iterator
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_AMINO_ACID_ATOMS = ["N", "CA", "C", "O"]
DEFAULT_RNA_ATOMS = ["C1'", "C2'", "C3'", "C4'", "O4'"]
DEFAULT_DNA_ATOMS = ["C1'", "C2'", "C3'", "C4'", "O4'"]
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
RNA_RESIDUES = ["A", "C", "G", "U", "N"]
DNA_RESIDUES = ["A", "C", "G", "T", "N"]


def sliding_window_iterator(
    length: int, window_width: int, step_size: int
) -> Iterator[np.ndarray]:
    """Generator for a 1-dimensional sliding window of indexes.

    Each iteration yields a long tensor of indexes with the same length as the window
    width. The starting index is incremented by the step size each iteration.
    Note that the last window may be shorter than the window width.

    Parameters
    ----------
    length : int
        Length of the sequence to iterate over.
    window_width : int
        Width of the sliding window.
    step_size : int
        Step size for incrementing the window.

    Yields
    ------
    torch.Tensor
        Tensor of indexes for the current window.

    Example
    -------
    >>> for window in sliding_window_iterator(10, 3, 2):
    ...     print(window)
    [0 1 2]
    [2 3 4]
    [4 5 6]
    [6 7 8]
    [8 9]
    """
    for i in range(0, length - window_width + step_size, step_size):
        yield np.arange(i, min(i + window_width, length))


class BaseTemplateIterator(BaseModel):
    """Base class for defining template iterator configurations.

    Attributes
    ----------
    num_residues_removed : int
        Number of residues to remove from the structure at each step.
    residue_increment : int
        Number of residues to increment the removal by each iteration.
    residue_types : list[str]
        Types of residues to target for removal. Options are 'amino_acid', 'rna', 'dna'.
    amino_acid_atoms : list[str], optional
        List of atom type labels (in the PDB file) to remove from amino acid residues.
        Default is ['N', 'CA', 'C', 'O'].
    rna_atoms : list[str], optional
        List of atom type labels (in the PDB file) to remove from RNA residues.
        Default is ['C1', 'C2', 'C3', 'C4', 'O4'].
    dna_atoms : list[str], optional
        List of atom type labels (in the PDB file) to remove from DNA residues.
        Default is ['C1', 'C2', 'C3', 'C4', 'O4'].
    structure_df : pd.DataFrame
        Underlying Pandas DataFrame for the PDB model.
    """

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    num_residues_removed: Annotated[int, Field(gt=0)]
    residue_increment: Annotated[int, Field(gt=0)]

    residue_types: list[Literal["amino_acid", "rna", "dna"]]
    amino_acid_atoms: list[str] = DEFAULT_AMINO_ACID_ATOMS
    rna_atoms: list[str] = DEFAULT_RNA_ATOMS
    dna_atoms: list[str] = DEFAULT_DNA_ATOMS

    structure_df: pd.DataFrame  # NOTE: Comes from Simulator object

    # Need to setup a data structure to map from chain, residue paris to atom indices
    # and the types of atoms to remove (this is already set up, but want to simplify).

    @field_validator("residue_types")  # type: ignore
    def _validate_residue_types(cls, v):
        if not v:
            raise ValueError("At least one residue type must be specified.")
        return v

    @abstractmethod
    def atom_idx_iter(self, inverted: bool = False) -> Iterator[torch.Tensor]:
        """Generator for iterating over atom indexes to keep in each structure.

        Parameters
        ----------
        inverted : bool
            If 'True', return the indexes of atoms to remove rather than keep.

        Yields
        ------
        torch.Tensor
            Tensor of indexes for the atoms to remove.
        """
        raise NotImplementedError


class RandomTemplateIterator(BaseTemplateIterator):
    """Template iterator for removing random atoms from a pdb structure.

    Attributes
    ----------
    type : Literal["random"]
        Discriminator field for differentiating between template iterator types.
    coherent_removal : bool
        If 'True', remove atoms from residues in order. For example, would remove atoms
        from residue [i, i+1, i+2, ...] rather than random indices. Default is 'True'.
    """

    type: Literal["random"] = "random"
    coherent_removal: bool = True


class ResidueTemplateIterator(BaseTemplateIterator):
    """Template iterator for removing chunks of atoms from residues from a structure.

    NOTE: If you want to set a fixed chain order (e.g. for reproducibility), you can
    use the built-in function (TODO).

    Attributes
    ----------
    type : Literal["residue"]
        Discriminator field for differentiating between template iterator types.
    randomize_chain_order : bool
        If 'True', randomize the order of chains in the structure. Default is 'False'.
    """

    type: Literal["residue"] = "residue"
    randomize_chain_order: bool = False

    _chain_order: list[str]

    def __init__(self, **data: Any):
        super().__init__(**data)

        # The unique method should retain default order
        self._chain_order = self.structure_df["chain"].unique()
        if self.randomize_chain_order:
            np.random.shuffle(self._chain_order)

    def set_chain_order(self, chain_order: list[str]) -> None:
        """Set the order of chains to iterate over.

        Parameters
        ----------
        chain_order : list[str]
            List of chain identifiers, in desired order, to use when iterating
            over the structure.

        Raises
        ------
        ValueError
            If the chain order does not contain all chains in the structure.

        Returns
        -------
        None
        """
        # Check that all the chains are present in the chain_order list
        if set(chain_order) != set(self.structure_df["chain"].unique()):
            raise ValueError("Chain order must contain all chains in the structure.")

        self._chain_order = chain_order

    def chain_residue_pairs(self) -> list[tuple[str, int]]:
        """Get the (chain, residue) pairs for the structure.

        NOTE: The returned pairs are unique and in the desired chain order. If you
        want to randomize or otherwise set the the order, must be done before calling
        this method.

        # This method is necessary because a PDB file may be malformed and have
        # duplicate chain identifiers. This method should handle the case where there
        # might be duplicate chain identifiers.

        Returns
        -------
        list[tuple[str, int]]
            List of (chain, residue_id) pairs in the structure.
        """
        # Determine which residues to keep based on the residue types
        keep_residues = []
        if "amino_acid" in self.residue_types:
            keep_residues.extend(AMINO_ACID_RESIDUES)
        if "rna" in self.residue_types:
            keep_residues.extend(RNA_RESIDUES)
        if "dna" in self.residue_types:
            keep_residues.extend(DNA_RESIDUES)

        # Subset the df to only residues that match the desired residue types
        subset_df = self.structure_df[self.structure_df["residue"].isin(keep_residues)]

        # Chunk the df into groups based on chain and re-stich together in order
        # This will ensure that the chain order is respected
        df_list = []
        for chain in self._chain_order:
            df_list.append(subset_df[subset_df["chain"] == chain])
        ordered_df = pd.concat(df_list)

        # Find unique (chain, residue_id) pairs, in order
        unique_chain_res_id = ordered_df[["chain", "residue_id"]].drop_duplicates()

        return unique_chain_res_id.to_records(index=False).tolist()  # type: ignore

    def atom_idx_iter(self, inverted: bool = False) -> Iterator[torch.Tensor]:
        """Generator for iterating over atom indexes to keep in each structure.

        Parameters
        ----------
        inverted : bool
            If 'True', return the indexes of atoms to remove rather than keep.

        Yields
        ------
        torch.Tensor
            Tensor of indexes for the atoms to remove.
        """
        remove_atoms = []
        if "amino_acid" in self.residue_types:
            remove_atoms.extend(self.amino_acid_atoms)
        if "rna" in self.residue_types:
            remove_atoms.extend(self.rna_atoms)
        if "dna" in self.residue_types:
            remove_atoms.extend(self.dna_atoms)

        # Get the unique chain, residue pairs in order
        chain_res_pairs = self.chain_residue_pairs()
        chains = np.array([chain for chain, _ in chain_res_pairs])
        residues = np.array([residue for _, residue in chain_res_pairs])

        window_iter = sliding_window_iterator(
            length=len(chain_res_pairs),
            window_width=self.num_residues_removed,
            step_size=self.residue_increment,
        )

        # Iterate over the chain, residue pairs and yield the atom indexes
        for window in window_iter:
            chains_window = chains[window]
            residues_window = residues[window]

            # Merge the DataFrame to keep only positions where the chain and residue
            # pairs match the current window
            merge_df = pd.DataFrame(
                {"chain": chains_window, "residue_id": residues_window}
            )
            df_window = self.structure_df.merge(merge_df)

            # Get the atom indexes for the current window (atoms that should be removed)
            atom_idxs = df_window[df_window["atom"].isin(remove_atoms)].index

            if inverted:
                atom_idxs = np.setdiff1d(np.arange(len(self.structure_df)), atom_idxs)

            yield torch.tensor(atom_idxs)


# class PrecalculatedVolumesTemplateIterator(BaseTemplateIterator):
#     """Template iterator for iterating over a set of precalculated volumes."""

#     pass
#     _type: Literal["precalculated-list"] = "precalculated-list"
