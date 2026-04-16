"""Module for computing expected overlap between structures."""

from dataclasses import dataclass

import torch


@dataclass
class AtomicStructure:
    """Container for 3D atomic structure data.

    Parameters
    ----------
    positions_zyx : torch.Tensor
        Tensor of shape (N, 3) containing the atomic positions in ZYX order.
    a : torch.Tensor
        Tensor of shape (N, 5) for scattering amplitude coefficients.
    b : torch.Tensor
        Tensor of shape (N, 5) for scattering width coefficients.
    B : torch.Tensor
        Tensor of shape (N,) for atomic B-factors.
    """

    positions_zyx: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    B: torch.Tensor

    def to(self, device: torch.device) -> "AtomicStructure":
        """Move all tensors to the specified device."""
        return AtomicStructure(
            positions_zyx=self.positions_zyx.to(device),
            a=self.a.to(device),
            b=self.b.to(device),
            B=self.B.to(device),
        )


def compute_q(b: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Gaussian variance per atom (q_i = 4pi / (B + b_i)).

    Parameters
    ----------
    b : torch.Tensor
        Tensor of shape (N, :) for electron scattering width coefficients.
    B : torch.Tensor
        Tensor of shape (N,) for atomic B- (Debye-Waller) factors.

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, :) containing the total Gaussian variance for each atom.
    """
    return 4 * torch.pi / (B.unsqueeze(-1) + b)


def projection_overlap_naive(
    structure_A: AtomicStructure,
    structure_B: AtomicStructure,
    rotation_matrices: torch.Tensor,  # shape (M, 3, 3)
) -> torch.Tensor:  # shape (M,)
    """A naive looped implementation of the projection overlap calculation.

    Computes the 2D projection overlap integral between two structures
    for each rotation, using the full pairwise interaction model.

    Parameters
    ----------
    structure_A : AtomicStructure
        First atomic structure.
    structure_B : AtomicStructure
        Second atomic structure, can be the same as structure_A.
    rotation_matrices : torch.Tensor
        Tensor of shape (M, 3, 3) containing M rotation matrices.

    Returns
    -------
    torch.Tensor
        Tensor of shape (M,) containing the overlap values for each rotation.
    """
    device = rotation_matrices.device
    dtype = rotation_matrices.dtype
    M = rotation_matrices.shape[0]

    q_A = compute_q(structure_A.b, structure_A.B)
    q_B = compute_q(structure_B.b, structure_B.B)

    sums = torch.zeros(M, device=device, dtype=dtype)

    for m in range(M):
        R = rotation_matrices[m]

        # Rotate and project atom centers for both structures, project onto XY plane
        proj_A = (structure_A.positions_zyx @ R.T)[..., :2]
        proj_B = (structure_B.positions_zyx @ R.T)[..., :2]

        dists = torch.cdist(proj_A, proj_B)
        dists_squared = dists**2

        # Accumulate over the 5x5 interacting Gaussian terms
        for k in range(5):
            for l in range(5):
                q_kl = q_A[:, k].unsqueeze(1) + q_B[:, l].unsqueeze(0)
                c_kl = structure_A.a[:, k].unsqueeze(1) * structure_B.a[:, l].unsqueeze(0)  # fmt: skip

                coeff = c_kl / (2 * torch.pi * q_kl)
                exponent = -0.5 * dists_squared / q_kl
                sums[m] += torch.sum(coeff * torch.exp(exponent))

    return sums


@dataclass
class ProjectionOverlapCalculator:
    """GPU-friendly overlap calculator using substructure accumulation.

    Instead of materializing all pairs, we compute block-by-block for any interacting
    atoms and accumulate contributions. This trades pre-computation time into something
    more memory efficient.

    Attributes
    ----------
    structure_A : AtomicStructure
        First atomic structure.
    structure_B : AtomicStructure
        Second atomic structure, can be the same as structure_A.
    cutoff_distance : float
        Maximum distance in the projected plane to consider atom pairs as interacting.
    autocorrelation : bool
        If True, computes autocorrelation (structure_A == structure_B) and applies
        symmetry optimizations.
    atom_block_size_A : int
        Number of atoms in structure A to process in each block. By default, 1024.
    atom_block_size_B : int
        Number of atoms in structure B to process in each block. By default, 1024.
    """

    structure_A: AtomicStructure
    structure_B: AtomicStructure
    cutoff_distance: float = 24.0
    autocorrelation: bool = False
    atom_block_size_A: int = 1024
    atom_block_size_B: int = 1024

    def __post_init__(self):
        """Initialize precomputed atomic parameters."""
        self.q_A = compute_q(self.structure_A.b, self.structure_A.B)
        self.q_B = compute_q(self.structure_B.b, self.structure_B.B)

    @torch.no_grad()
    def evaluate(self, rotation_matrices: torch.Tensor) -> torch.Tensor:
        """Evaluate overlap for multiple rotations without materializing all pairs.

        Parameters
        ----------
        rotation_matrices : torch.Tensor
            Shape (M, 3, 3) rotation matrices.

        Returns
        -------
        torch.Tensor
            Shape (M,) overlap values.
        """
        device = rotation_matrices.device
        dtype = rotation_matrices.dtype
        M = rotation_matrices.shape[0]

        num_atoms_A = self.structure_A.positions_zyx.shape[0]
        num_atoms_B = self.structure_B.positions_zyx.shape[0]
        cutoff_squared = self.cutoff_distance**2

        sums = torch.zeros(M, device=device, dtype=dtype)

        for m in range(M):
            R = rotation_matrices[m]

            # Rotate and project all atoms at once
            proj_A = (self.structure_A.positions_zyx @ R.T)[..., :2]
            proj_B = (self.structure_B.positions_zyx @ R.T)[..., :2]

            # Accumulate contribution from each block pair
            for i_start in range(0, num_atoms_A, self.atom_block_size_A):
                i_end = min(i_start + self.atom_block_size_A, num_atoms_A)
                proj_A_block = proj_A[i_start:i_end]  # (B_A, 2)

                for j_start in range(0, num_atoms_B, self.atom_block_size_B):
                    j_end = min(j_start + self.atom_block_size_B, num_atoms_B)
                    proj_B_block = proj_B[j_start:j_end]  # (B_B, 2)

                    # Compute block-level overlap contribution
                    block_contrib = self._compute_block_overlap(
                        i_start,
                        i_end,
                        j_start,
                        j_end,
                        proj_A_block,
                        proj_B_block,
                        cutoff_squared,
                    )
                    sums[m] += block_contrib

        return sums

    @torch.no_grad()
    def _compute_block_overlap(
        self,
        i_start: int,
        i_end: int,
        j_start: int,
        j_end: int,
        proj_A_block: torch.Tensor,  # (B_A, 2)
        proj_B_block: torch.Tensor,  # (B_B, 2)
        cutoff_squared: float,
    ) -> torch.Tensor:
        """Compute overlap contribution for a single block pair.

        Parameters
        ----------
        i_start, i_end : int
            Atom index range in structure A
        j_start, j_end : int
            Atom index range in structure B
        proj_A_block : torch.Tensor
            Projected positions for A atoms in this block
        proj_B_block : torch.Tensor
            Projected positions for B atoms in this block
        cutoff_squared : float
            Cutoff distance squared

        Returns
        -------
        torch.Tensor
            Scalar contribution to overlap.
        """
        device = proj_A_block.device
        dtype = proj_A_block.dtype

        # Compute pairwise distances for this block
        dists_sq = torch.cdist(proj_A_block, proj_B_block) ** 2

        # Skip blocks entirely outside cutoff
        if cutoff_squared < float("inf") and dists_sq.min() > cutoff_squared:
            return torch.tensor(0.0, device=device, dtype=dtype)

        # Find active pairs (within cutoff)
        if cutoff_squared < float("inf"):
            mask = dists_sq < cutoff_squared
            if not mask.any():
                return torch.tensor(0.0, device=device, dtype=dtype)
        else:
            mask = torch.ones_like(dists_sq, dtype=torch.bool)

        # Get local indices of active pairs
        local_i, local_j = torch.nonzero(mask, as_tuple=True)

        # Convert to global atom indices
        global_i = local_i + i_start
        global_j = local_j + j_start

        # Extract atomic parameters for active pairs
        q_A_pairs = self.q_A[global_i]  # (P, 5)
        q_B_pairs = self.q_B[global_j]  # (P, 5)
        a_A_pairs = self.structure_A.a[global_i]  # (P, 5)
        a_B_pairs = self.structure_B.a[global_j]  # (P, 5)

        # Compute all 25 (k, l) pair contributions at once
        # Shapes: (P, 1, 5) + (P, 5, 1) -> (P, 5, 5)
        q_kl = q_A_pairs.unsqueeze(2) + q_B_pairs.unsqueeze(1)  # (P, 5, 5)
        c_kl = a_A_pairs.unsqueeze(2) * a_B_pairs.unsqueeze(1)  # (P, 5, 5)

        # Compute coefficients and exponents
        coeff = c_kl / (2 * torch.pi * q_kl)  # (P, 5, 5)

        # Get distances for active pairs
        dists_sq_active = dists_sq[mask]  # (P,)
        exponent = (
            -0.5 * dists_sq_active.unsqueeze(-1).unsqueeze(-1) / q_kl
        )  # (P, 5, 5)

        # Sum all contributions
        contribution = torch.sum(coeff * torch.exp(exponent))

        return contribution
