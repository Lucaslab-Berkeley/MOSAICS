"""Manager class for running MOSAICS."""

import warnings
from typing import Literal, Union

import mmdf
import torch
import tqdm
import yaml  # type: ignore
from pydantic import BaseModel
from torch_fourier_slice import project_3d_to_2d
from ttsim3d.models import Simulator

from .cross_correlation_core import (
    alternate_template_statistics,
    cross_correlate_particle_stack,
    project_filtered_templates,
)
from .mosaics_dataset import MosaicsDataset
from .mosaics_result import AlternateTemplateResult, MosaicsResult
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


def _get_full_length_projections(
    default_volume: torch.Tensor,
    rotation_matrices: torch.Tensor,
    projective_filters: torch.Tensor,
    batch_size: int = 2048,
) -> torch.Tensor:
    """Calculate the full-length projections for the default template.

    NOTE: Kept for backwards compatibility only. These projections are computed from
    a padded volume and so differ slightly (interpolation) from the templates used
    for cross-correlation. Use `project_filtered_templates` instead.
    """
    full_length_projections = torch.zeros(
        (
            rotation_matrices.shape[0],
            default_volume.shape[-2],
            default_volume.shape[-1],
        ),
        device=default_volume.device,
    )

    for i in range(0, rotation_matrices.shape[0], batch_size):
        batch_rotation_matrices = rotation_matrices[i : i + batch_size]
        batch_projective_filters = projective_filters[i : i + batch_size]

        batch_projections = project_3d_to_2d(
            volume=default_volume, rotation_matrices=batch_rotation_matrices
        )
        batch_projections_dft = torch.fft.rfftn(batch_projections, dim=(-2, -1))
        batch_projections_dft *= batch_projective_filters
        batch_projections_dft *= -1
        batch_projections = torch.fft.irfftn(batch_projections_dft, dim=(-2, -1))

        full_length_projections[i : i + batch_size] = batch_projections

    return full_length_projections


def _volume_to_template_dft(volume: torch.Tensor) -> torch.Tensor:
    """Real-Fourier transform a volume into the layout used for slice extraction."""
    volume = torch.fft.fftshift(volume)
    volume_dft = torch.fft.rfftn(volume, dim=(-3, -2, -1))
    return torch.fft.fftshift(volume_dft, dim=(-3, -2))


class MosaicsManager(BaseModel):
    """Class for importing, running, and exporting MOSAICS program data.

    Attributes
    ----------
    simulator : Simulator
        Instance of Simulator model from ttsim3d package. Holds the pdb file and
        associated atom positions, bfactors, etc. for simulating a 3D volume.
    template_iterator : BaseTemplateIterator
        Iteration configuration model for describing how to iterate over the reference
        structure. Should be an instance of a subclass of BaseTemplateIterator.
    dataset : MosaicsDataset
        Dataset object that handles processing, loading, and caching of input data.
    sim_removed_atoms_only : bool
        When True, only re-simulate the removed atoms from the alternate template and
        subtract the alternate volume from the default volume. When False, simulate the
        entire alternate template and subtract the alternate volume from the default.
        Simulating only the removed atoms is generally faster. Default is True.
    particle_positions_are_centered : bool
        If True, then assume that the extracted boxes of particle images coming from
        'dataset.particle_images' will produce correlogram peak at central pixel.
        If False (typical), then determine the (x, y) position of each correlogram peak
        before proceeding. Default is False. NOTE: When this is set to False,
        the extracted box size should be larger than the maximum expected particle
        shift.
    """

    simulator: Simulator
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

    def _mosaics_inner_loop(
        self,
        particle_images: torch.Tensor,
        full_length_projections: torch.Tensor,
        rotation_matrices: torch.Tensor,
        projective_filters: torch.Tensor,
        default_volume: torch.Tensor,
        atom_indices: torch.Tensor,
        device: torch.device,
        batch_size: int = 2048,
    ) -> dict[str, torch.Tensor]:
        """Inner loop function for running the MOSAICS program.

        Parameters
        ----------
        particle_images : torch.Tensor
            Pre-processed and normalized particle images *in real space*.
        full_length_projections : torch.Tensor
            The filtered default template projections from `project_filtered_templates`.
        rotation_matrices : torch.Tensor
            The rotation matrices for the orientations of each particle.
        projective_filters : torch.Tensor
            The projection filters for each particle image.
        default_volume : torch.Tensor
            The default (full-length) simulated structure volume to compare against.
        atom_indices : torch.Tensor
            Atom indices passed to the simulator. These are the removed atoms when
            'sim_removed_atoms_only' is True, otherwise the kept atoms.
        device : torch.device
            The device to use for the computation. Should be either a CPU or GPU device.
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
        simulated_volume = self.simulator.run(
            device=str(device), atom_indices=atom_indices
        )

        # When self.sim_removed_atoms_only is set, only the removed atoms are
        # simulated; otherwise only the kept atoms are simulated.
        if self.sim_removed_atoms_only:
            removed_volume = simulated_volume
            alternate_volume = default_volume - simulated_volume
        else:
            removed_volume = default_volume - simulated_volume
            alternate_volume = simulated_volume

        return alternate_template_statistics(
            particle_stack_images=particle_images,
            default_projections=full_length_projections,
            alternate_template_dft=_volume_to_template_dft(alternate_volume),
            removed_template_dft=_volume_to_template_dft(removed_volume),
            rotation_matrices=rotation_matrices,
            projective_filters=projective_filters,
            batch_size=batch_size,
        )

    def run_mosaics(
        self,
        gpu_id: Union[Literal["cpu"], int],
        batch_size: int = 2048,
        cache_path: Union[None, str] = None,
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
            used.
        """
        if gpu_id == "cpu":
            device = torch.device("cpu")
        elif isinstance(gpu_id, int) and gpu_id >= 0:
            device = torch.device(f"cuda:{gpu_id}")
        else:
            raise ValueError(
                f"Invalid gpu_id: {gpu_id}. Must be 'cpu' or a non-negative integer."
            )

        # Load or process the dataset
        if cache_path is not None:
            self.dataset.load_from_cache(cache_path, device)
        else:
            self.dataset.process_inputs(device, self.simulator)

        # Extract processed variables from dataset
        rotation_matrices = self.dataset.rotation_matrices
        projective_filters = self.dataset.projective_filters
        particle_images = self.dataset.particle_images
        default_template = self.dataset.template_volume
        default_template_dft = self.dataset.template_volume_dft

        #####################################################################
        ### 1. Calculate default (full length) cross corr and projections ###
        #####################################################################

        # Filtered default template projections t, identical to the templates
        # correlated against the particle images (needed for the Gram entries).
        full_length_projections = project_filtered_templates(
            template_dft=default_template_dft,
            rotation_matrices=rotation_matrices,
            projective_filters=projective_filters,
            batch_size=batch_size,
        )

        default_cc, _ = cross_correlate_particle_stack(
            particle_stack_images=particle_images,
            perfect_projection_images=None,  # Don't need auto-correlation
            template_dft=default_template_dft,
            rotation_matrices=rotation_matrices,
            projective_filters=projective_filters,
            batch_size=batch_size,
        )

        default_scattering_potential = (
            self.template_iterator.get_template_scattering_potential(None)
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
                    stacklevel=2,
                )
            else:
                orig_template_size = self.dataset.particle_stack.original_template_size
                _, particle_images = _center_images_by_correlations(
                    cross_correlation=default_cc,
                    particle_images=particle_images,
                    original_template_size=orig_template_size,
                )

        default_cc, _ = cross_correlate_particle_stack(
            particle_stack_images=particle_images,
            perfect_projection_images=None,
            template_dft=default_template_dft,
            rotation_matrices=rotation_matrices,
            projective_filters=projective_filters,
            batch_size=batch_size,
        )

        # Default template autocorrelation s_t = ||t||^2 (pose-dependent only)
        default_overlap = torch.sum(full_length_projections**2, dim=(-2, -1))

        ######################################################
        ### 3. Iteration over alternate (truncated) models ###
        ######################################################

        num_iters = self.template_iterator.num_alternate_structures

        # NOTE: When the inverted flag is set to True, the iterator will return the
        # indices of the atoms that should NOT be removed. This is opposite of the
        # the 'sim_removed_atoms_only' flag.
        invert_iterator = not self.sim_removed_atoms_only
        alternate_template_iterator = self.template_iterator.alternate_template_iter(
            invert_iterator
        )

        alternate_template_results = []
        for chains, residues, atom_indices in tqdm.tqdm(
            alternate_template_iterator,
            total=num_iters,
            desc="Iterating over alternate models",
        ):
            if len(atom_indices) == 0:
                warnings.warn(
                    "No atoms to remove for this iteration. Skipping calculation.",
                    RuntimeWarning,
                    stacklevel=2,
                )
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
                    rotation_matrices=rotation_matrices,
                    projective_filters=projective_filters,
                    default_volume=default_template,
                    atom_indices=atom_indices,
                    device=device,
                    batch_size=batch_size,
                )

            stats_np = {key: value.cpu().numpy() for key, value in stats.items()}
            alternate_scattering_potential = (
                self.template_iterator.get_template_scattering_potential(atom_indices)
            )

            this_result = AlternateTemplateResult(
                chain_ids=chains,
                residue_ids=residues,
                removed_atom_indices=atom_indices.cpu().numpy(),
                sim_removed_atoms_only=self.sim_removed_atoms_only,
                scattering_potential_full_length=default_scattering_potential,
                scattering_potential_alternate=alternate_scattering_potential,
                **stats_np,
            )
            alternate_template_results.append(this_result)

        return MosaicsResult(
            default_cross_correlation=default_cc.cpu().numpy(),
            default_overlap=default_overlap.cpu().numpy(),
            template_iterator_config=self.template_iterator.model_dump(),
            sim_removed_atoms_only=self.sim_removed_atoms_only,
            alternate_template_results=alternate_template_results,
        )
