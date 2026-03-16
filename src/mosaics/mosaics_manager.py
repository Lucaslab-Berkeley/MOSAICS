"""Manager class for running MOSAICS."""

import warnings
from typing import Literal, Union

import mmdf
import numpy as np
import roma
import torch
import tqdm
import yaml  # type: ignore
from leopard_em.pydantic_models.config import PreprocessingFilters
from leopard_em.pydantic_models.data_structures import ParticleStack
from leopard_em.pydantic_models.utils import (
    _setup_ctf_kwargs_from_particle_stack,
    calculate_ctf_filter_stack_full_args,
    setup_images_filters_particle_stack,
)
from pydantic import BaseModel
from ttsim3d.models import Simulator

from .cross_correlation_core import cross_correlate_particle_stack
from .mosaics_dataset import MosaicsDataset
from .mosaics_result import AlternateTemplateResult, MosaicsResult
from .template_iterator import BaseTemplateIterator, instantiate_template_iterator


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
    """

    simulator: Simulator
    template_iterator: BaseTemplateIterator
    # particle_stack: ParticleStack
    # preprocessing_filters: PreprocessingFilters
    dataset: MosaicsDataset
    sim_removed_atoms_only: bool = True

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
        data["template_iterator"]["structure_df"] = pdb_df

        # Create the template iterator using the factory method
        template_iterator = instantiate_template_iterator(data["template_iterator"])
        data["template_iterator"] = template_iterator

        return cls(**data)

    def _mosaics_inner_loop(
        self,
        particle_images: torch.Tensor,
        rotation_matrices: torch.Tensor,
        projective_filters: torch.Tensor,
        default_volume: torch.Tensor,
        atom_indices: torch.Tensor,
        device: torch.device,
        batch_size: int = 2048,
    ) -> torch.Tensor:
        """Inner loop function for running the MOSAICS program.

        Parameters
        ----------
        particle_images : torch.Tensor
            Pre-processed and normalized particle images *in real space*.
        rotation_matrices : torch.Tensor
            The rotation matrices for the orientations of each particle.
        projective_filters : torch.Tensor
            The projection filters for each particle image.
        default_volume : torch.Tensor
            The default (full-length) simulated structure volume to compare against.
        atom_indices : torch.Tensor
            Which atoms should be removed from the template for the alternate model.
        device : torch.device
            The device to use for the computation. Should be either a CPU or GPU device.
        batch_size : int, optional
            The batch size to use for the cross-correlation calculations. Default is
            2048.

        Returns
        -------
        torch.Tensor
            The cross-correlation values for the alternate template. Shape of (N,).
        """
        alternate_volume = self.simulator.run(
            device=str(device), atom_indices=atom_indices
        )

        # Subtract the alternate_volume from the default_volume if
        # self.sim_removed_atoms_only is set.
        # This is because, when inverted, only the atoms which should be
        # removed get simulated rather than the atoms which should be kept.
        if self.sim_removed_atoms_only:
            alternate_volume = default_volume - alternate_volume

        alternate_volume = torch.fft.fftshift(alternate_volume)
        alternate_volume_dft = torch.fft.rfftn(alternate_volume, dim=(-3, -2, -1))
        alternate_volume_dft = torch.fft.fftshift(alternate_volume_dft, dim=(-3, -2))

        # Recalculate the cross-correlation with the alternate model
        # and take the maximum value over space
        alternate_cc = cross_correlate_particle_stack(
            particle_stack_images=particle_images,
            template_dft=alternate_volume_dft,
            rotation_matrices=rotation_matrices,
            projective_filters=projective_filters,
            batch_size=batch_size,
        )

        return alternate_cc

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

        #####################################################
        ### 1. Calculate default (full length) cross corr ###
        #####################################################

        default_cc = cross_correlate_particle_stack(
            particle_stack_images=particle_images,
            template_dft=default_template_dft,
            rotation_matrices=rotation_matrices,
            projective_filters=projective_filters,
            batch_size=batch_size,
        )

        default_scattering_potential = self.template_iterator.get_template_scattering_potential(None)

        ######################################################
        ### 2. Iteration over alternate (truncated) models ###
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
                alternate_cc = default_cc
                alternate_scattering_potential = default_scattering_potential

            else:
                alternate_cc = self._mosaics_inner_loop(
                    particle_images=particle_images,
                    rotation_matrices=rotation_matrices,
                    projective_filters=projective_filters,
                    default_volume=default_template,
                    atom_indices=atom_indices,
                    device=device,
                )
            alternate_cc = alternate_cc.cpu().numpy()
            alternate_scattering_potential = self.template_iterator.get_template_scattering_potential(
                atom_indices
            )

            this_result = AlternateTemplateResult(
                cross_correlation=alternate_cc,
                chain_ids=chains,
                residue_ids=residues,
                removed_atom_indices=atom_indices.cpu().numpy(),
                sim_removed_atoms_only=self.sim_removed_atoms_only,
                scattering_potential_full_length=default_scattering_potential,
                scattering_potential_alternate=alternate_scattering_potential,
            )
            alternate_template_results.append(this_result)

        return MosaicsResult(
            default_cross_correlation=default_cc.cpu().numpy(),
            template_iterator_config=self.template_iterator.model_dump(),
            sim_removed_atoms_only=self.sim_removed_atoms_only,
            alternate_template_results=alternate_template_results,
        )
