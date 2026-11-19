"""Manager class for running MOSAICS."""

import warnings
from typing import Literal, Union

import mmdf
import numpy as np
import torch
import tqdm
import yaml  # type: ignore
from pydantic import BaseModel

from .cross_correlation_core import (
    alternate_template_statistics,
    cross_correlate_particle_stack,
    project_filtered_templates,
)
from .mosaics_dataset import MosaicsDataset
from .mosaics_result import AlternateTemplateResult, MosaicsResult
from .projection import MosaicsSimulator, TemplateProjector
from .template_iterator import BaseTemplateIterator, instantiate_template_iterator

SAME_BOX_SIZE_WARNING = (
    "The extracted box size is the as projection size, but requested particle shifts "
    "to be determined. Cannot find shifts with the same sizes. Please set "
    "'extracted_box_size' to be ~15-20 pct larger than 'original_template_size'."
)


def _center_images_by_correlations(
    cross_correlation: torch.Tensor,  # (N, H - h, W - w)
    particle_images: torch.Tensor,  # (N, H, W)
    original_template_size: tuple[int, int],  # (h, w)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Center and crop particle images by maximum cross-correlation peak."""
    H, W = particle_images.shape[-2], particle_images.shape[-1]
    h, w = original_template_size

    cc = cross_correlation.view(cross_correlation.shape[0], -1)
    max_cc_values, max_cc_indices = torch.max(cc, dim=1)
    max_xy_indices = torch.unravel_index(max_cc_indices, cross_correlation.shape[1:])

    # Crop to same shape as projection around the peak
    centered_images = torch.zeros(
        (particle_images.shape[0], h, w), device=particle_images.device
    )

    ccg_center_y = (H - h) // 2
    ccg_center_x = (W - w) // 2
    for i in range(particle_images.shape[0]):
        x_pos, y_pos = max_xy_indices[0][i], max_xy_indices[1][i]

        shift_y = y_pos - ccg_center_y
        shift_x = x_pos - ccg_center_x

        y_start = ccg_center_y + shift_y
        y_end = y_start + h
        x_start = ccg_center_x + shift_x
        x_end = x_start + w

        centered_images[i] = particle_images[i, x_start:x_end, y_start:y_end]

    # Renormalize the centered images and correlation values based on new shape
    factor = ((h * w) / (H * W)) ** 0.5
    centered_images *= factor
    max_cc_values *= factor

    return max_cc_values, centered_images


class MosaicsManager(BaseModel):
    """Class for importing, running, and exporting MOSAICS program data.

    Attributes
    ----------
    simulator : MosaicsSimulator
        Instance of MosaicsSimulator, the ttsim3d Simulator model extended with the
        settings of the analytic projections (e.g. 'stencil_radius'). Holds the pdb
        file and
        associated atom positions, bfactors, etc. along with the simulation settings
        (dose weighting, MTF, upsampling). Template projections are computed
        analytically from these atoms (see `mosaics.projection.TemplateProjector`);
        no 3D volume is simulated.
    template_iterator : BaseTemplateIterator
        Iteration configuration model for describing how to iterate over the reference
        structure. Should be an instance of a subclass of BaseTemplateIterator.
    dataset : MosaicsDataset
        Dataset object that handles processing, loading, and caching of input data.
    sim_removed_atoms_only : bool
        Whether the template iterator returns the removed atoms (True) or the kept
        atoms (False) of each alternate template. This only affects how the atom
        indices are interpreted and stored: the projections are linear in the atoms,
        so only the removed atoms are ever projected. Default is True.
    particle_positions_are_centered : bool
        If True, then assume that the extracted boxes of particle images coming from
        'dataset.particle_images' will produce correlogram peak at central pixel.
        If False (typical), then determine the (x, y) position of each correlogram peak
        before proceeding. Default is False. NOTE: When this is set to False,
        the extracted box size should be larger than the maximum expected particle
        shift.
    """

    simulator: MosaicsSimulator
    template_iterator: BaseTemplateIterator
    dataset: MosaicsDataset
    sim_removed_atoms_only: bool = True
    particle_positions_are_centered: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "MosaicsManager":
        """Create a MosaicsManager instance from a YAML file.

        Parameters
        ----------
        yaml_path : str
            Path to the YAML file containing the configuration for the MosaicsManager.

        Returns
        -------
        MosaicsManager
            Instance of MosaicsManager created from the YAML file.
        """
        with open(yaml_path) as yaml_f:
            data = yaml.load(yaml_f, Loader=yaml.SafeLoader)

        # Load the pdb file from the Simulator into a DataFrame
        pdb_df = mmdf.read(data["simulator"]["pdb_filepath"])
        # removes hydrogens from pdb_df if option is enabled
        if data["simulator"].get("remove_hydrogens", True):
            pdb_df = pdb_df[pdb_df["element"] != "H"].reset_index()
        data["template_iterator"]["structure_df"] = pdb_df

        # Create the template iterator using the factory method
        template_iterator = instantiate_template_iterator(data["template_iterator"])
        data["template_iterator"] = template_iterator

        return cls(**data)

    def _removed_atom_indices(
        self, atom_indices: torch.Tensor, num_atoms: int
    ) -> torch.Tensor:
        """Indices of the removed atoms given the template iterator's atom indices."""
        atom_indices = torch.as_tensor(atom_indices).long()

        if self.sim_removed_atoms_only:
            return atom_indices

        is_removed = torch.ones(num_atoms, dtype=torch.bool)
        is_removed[atom_indices] = False

        return torch.nonzero(is_removed).squeeze(-1)

    def _mosaics_inner_loop(
        self,
        particle_images: torch.Tensor,
        full_length_projections: torch.Tensor,
        projector: TemplateProjector,
        rotation_matrices: torch.Tensor,
        projective_filters: torch.Tensor,
        atom_indices: torch.Tensor,
        batch_size: int = 2048,
    ) -> dict[str, torch.Tensor]:
        """Inner loop function for running the MOSAICS program.

        Parameters
        ----------
        particle_images : torch.Tensor
            Pre-processed and normalized particle images *in real space*.
        full_length_projections : torch.Tensor
            The filtered default template projections from `project_filtered_templates`.
        projector : TemplateProjector
            The analytic projector holding all template atoms.
        rotation_matrices : torch.Tensor
            The rotation matrices for the orientations of each particle.
        projective_filters : torch.Tensor
            The projection filters for each particle image.
        atom_indices : torch.Tensor
            Atom indices from the template iterator. These are the removed atoms when
            'sim_removed_atoms_only' is True, otherwise the kept atoms.
        batch_size : int, optional
            The batch size to use for the cross-correlation calculations. Default is
            2048.

        Returns
        -------
        dict[str, torch.Tensor]
            Per-particle statistics from `alternate_template_statistics`: the
            alternate cross-correlation c*, the overlap <t, t*>, and the Gram entries
            ||t*||^2, ||h||^2 and <t*, h> where h = t - t* is the removed structure.
        """
        return alternate_template_statistics(
            particle_stack_images=particle_images,
            default_projections=full_length_projections,
            projector=projector,
            removed_atom_indices=self._removed_atom_indices(
                atom_indices, projector.num_atoms
            ),
            rotation_matrices=rotation_matrices,
            projective_filters=projective_filters,
            batch_size=batch_size,
        )

    def _process_particle_chunk(
        self,
        inputs: dict[str, torch.Tensor],
        projector: TemplateProjector,
        alternates: list[tuple],
        batch_size: int,
        desc: str,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, np.ndarray]]]:
        """Default and alternate template statistics for one chunk of particles.

        Parameters
        ----------
        inputs : dict[str, torch.Tensor]
            The chunk's 'rotation_matrices', 'projective_filters' and
            'particle_images'. The images are popped from the dict so that the
            uncentered stack is freed once the images have been centered.
        projector : TemplateProjector
            The analytic projector holding all template atoms.
        alternates : list[tuple]
            The (chains, residues, atom_indices) of each alternate template, with the
            atom indices as returned by the template iterator (removed atoms when
            'sim_removed_atoms_only' is True, otherwise kept atoms).
        batch_size : int
            Number of particles processed at once.
        desc : str
            Progress bar description.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, list[dict[str, np.ndarray]]]
            The default cross-correlations, the default template autocorrelations
            and, per alternate template, the statistics of
            `alternate_template_statistics`.
        """
        rotation_matrices = inputs["rotation_matrices"]
        projective_filters = inputs["projective_filters"]
        particle_images = inputs.pop("particle_images")

        #####################################################################
        ### 1. Calculate default (full length) cross corr and projections ###
        #####################################################################

        # Filtered default template projections t, computed analytically from all
        # atoms. These are the templates correlated against the particle images.
        full_length_projections = project_filtered_templates(
            projector=projector,
            rotation_matrices=rotation_matrices,
            projective_filters=projective_filters,
            batch_size=batch_size,
        )

        default_cc = cross_correlate_particle_stack(
            particle_stack_images=particle_images,
            projections=full_length_projections,
            batch_size=batch_size,
        )

        ######################################################
        ### 2. Determine the position of correlogram peaks ###
        ######################################################

        if not self.particle_positions_are_centered:
            _box_size_is_same = (
                self.dataset.particle_stack.extracted_box_size
                == self.dataset.particle_stack.original_template_size
            )

            if _box_size_is_same:
                warnings.warn(
                    message=SAME_BOX_SIZE_WARNING,
                    category=RuntimeWarning,
                    stacklevel=3,
                )
            else:
                # Rebinding particle_images frees the uncentered stack
                orig_template_size = self.dataset.particle_stack.original_template_size
                _, particle_images = _center_images_by_correlations(
                    cross_correlation=default_cc,
                    particle_images=particle_images,
                    original_template_size=orig_template_size,
                )
                default_cc = cross_correlate_particle_stack(
                    particle_stack_images=particle_images,
                    projections=full_length_projections,
                    batch_size=batch_size,
                )

        # Default template autocorrelation s_t = ||t||^2 (pose-dependent only)
        default_overlap = torch.sum(full_length_projections**2, dim=(-2, -1))

        ######################################################
        ### 3. Iteration over alternate (truncated) models ###
        ######################################################

        alternate_stats = []
        for _, _, atom_indices in tqdm.tqdm(alternates, desc=desc):
            removed = self._removed_atom_indices(atom_indices, projector.num_atoms)
            if len(removed) == 0:
                # Nothing removed: t* = t and h = 0
                stats = {
                    "cross_correlation": default_cc,
                    "alternate_overlap": default_overlap,
                    "alternate_autocorrelation": default_overlap,
                    "removed_autocorrelation": torch.zeros_like(default_overlap),
                    "alternate_removed_overlap": torch.zeros_like(default_overlap),
                }
            else:
                stats = self._mosaics_inner_loop(
                    particle_images=particle_images,
                    full_length_projections=full_length_projections,
                    projector=projector,
                    rotation_matrices=rotation_matrices,
                    projective_filters=projective_filters,
                    atom_indices=atom_indices,
                    batch_size=batch_size,
                )
            alternate_stats.append({k: v.cpu().numpy() for k, v in stats.items()})

        return (
            default_cc.cpu().numpy(),
            default_overlap.cpu().numpy(),
            alternate_stats,
        )

    def run_mosaics(
        self,
        gpu_id: Union[Literal["cpu"], int],
        batch_size: int = 2048,
        cache_path: Union[None, str] = None,
        particle_chunk_size: Union[None, int] = None,
    ) -> MosaicsResult:
        """Run the MOSAICS program.

        Parameters
        ----------
        gpu_id : Union[Literal["cpu"], int]
            The GPU ID to use for the computation. Can either be the string "cpu" to
            use the CPU, or an integer specifying the GPU ID. All other values are
            invalid.
        batch_size : int, optional
            The batch size -- number of particle images to process at once -- to use
            for the cross-correlation calculations. The default is 2048.
        cache_path : Union[None, str], optional
            If a string is provided, then the path is assumed to contain a .npz file
            which contains pre-computed dataset variables. Useful when many experiments
            are being run in sequence. Default is None, which means no cache file is
            used. The cache is loaded onto the CPU and each chunk of particles is moved
            to the device as it is processed.
        particle_chunk_size : Union[None, int], optional
            Number of particles processed (pre-processed, projected and correlated
            against all alternate templates) at a time. Particles are independent, so
            the results do not depend on the chunk size, but memory use on the device
            scales with it. Default is None, which processes all particles at once.

        NOTE: Without a cache, the dataset is processed during the run and only holds
        the rotation matrices and projective filters of the last chunk afterwards (the
        particle images are released to save memory). To create a cache, call
        `dataset.process_inputs` followed by `dataset.save_to_cache` beforehand.
        """
        if gpu_id == "cpu":
            device = torch.device("cpu")
        elif isinstance(gpu_id, int) and gpu_id >= 0:
            device = torch.device(f"cuda:{gpu_id}")
        else:
            raise ValueError(
                f"Invalid gpu_id: {gpu_id}. Must be 'cpu' or a non-negative integer."
            )

        # The cache holds all particles; load it on the CPU and move chunks over
        if cache_path is not None:
            self.dataset.load_from_cache(cache_path, torch.device("cpu"))
            num_particles = int(self.dataset.rotation_matrices.shape[0])
        else:
            num_particles = self.dataset.num_particles

        if particle_chunk_size is None or particle_chunk_size <= 0:
            particle_chunk_size = num_particles
        particle_chunk_size = max(1, min(particle_chunk_size, num_particles))
        chunks = [
            slice(start, min(start + particle_chunk_size, num_particles))
            for start in range(0, num_particles, particle_chunk_size)
        ]

        projector = TemplateProjector.from_simulator(self.simulator, device)

        # NOTE: When the inverted flag is set to True, the iterator will return the
        # indices of the atoms that should NOT be removed. This is opposite of the
        # the 'sim_removed_atoms_only' flag. The alternates are materialized once so
        # every chunk of particles uses the same (possibly random) alternates.
        invert_iterator = not self.sim_removed_atoms_only
        alternates = list(
            self.template_iterator.alternate_template_iter(invert_iterator)
        )
        for j, (chains, _, atom_indices) in enumerate(alternates):
            if len(self._removed_atom_indices(atom_indices, projector.num_atoms)) == 0:
                chain_ids = sorted({str(chain) for chain in chains})
                warnings.warn(
                    f"Alternate template {j} (chains {chain_ids}) removes no atoms; "
                    "its statistics equal those of the default template.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        chunk_results = []
        for rows in chunks:
            if cache_path is not None:
                inputs = {
                    "rotation_matrices": self.dataset.rotation_matrices[rows],
                    "projective_filters": self.dataset.projective_filters[rows],
                    "particle_images": self.dataset.particle_images[rows],
                }
                inputs = {key: value.to(device) for key, value in inputs.items()}
            else:
                self.dataset.process_inputs(
                    device,
                    self.simulator,
                    particle_rows=rows if len(chunks) > 1 else None,
                )
                inputs = {
                    "rotation_matrices": self.dataset.rotation_matrices,
                    "projective_filters": self.dataset.projective_filters,
                    "particle_images": self.dataset.pop_particle_images(),
                }

            desc = "Iterating over alternate models"
            if len(chunks) > 1:
                desc += f" (particles {rows.start}-{rows.stop - 1} of {num_particles})"
            chunk_results.append(
                self._process_particle_chunk(
                    inputs, projector, alternates, batch_size, desc
                )
            )
            del inputs

        default_cc = np.concatenate([res[0] for res in chunk_results])
        default_overlap = np.concatenate([res[1] for res in chunk_results])

        default_scattering_potential = (
            self.template_iterator.get_template_scattering_potential(None)
        )
        alternate_template_results = []
        for j, (chains, residues, atom_indices) in enumerate(alternates):
            stats = {
                key: np.concatenate([res[2][j][key] for res in chunk_results])
                for key in chunk_results[0][2][j]
            }
            alternate_scattering_potential = (
                self.template_iterator.get_template_scattering_potential(atom_indices)
            )
            alternate_template_results.append(
                AlternateTemplateResult(
                    chain_ids=chains,
                    residue_ids=residues,
                    removed_atom_indices=np.asarray(atom_indices),
                    sim_removed_atoms_only=self.sim_removed_atoms_only,
                    scattering_potential_full_length=default_scattering_potential,
                    scattering_potential_alternate=alternate_scattering_potential,
                    **stats,
                )
            )

        return MosaicsResult(
            default_cross_correlation=default_cc,
            default_overlap=default_overlap,
            template_iterator_config=self.template_iterator.model_dump(),
            sim_removed_atoms_only=self.sim_removed_atoms_only,
            alternate_template_results=alternate_template_results,
        )
