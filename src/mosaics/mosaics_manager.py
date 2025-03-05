"""Manager class for running MOSAICS."""

import mmdf
import roma
import torch
import tqdm
import yaml  # type: ignore
from leopard_em.backend.core_refine_template import cross_correlate_particle_stack
from leopard_em.pydantic_models import ParticleStack, PreprocessingFilters
from pydantic import BaseModel
from torch_fourier_filter.ctf import calculate_ctf_2d
from ttsim3d.models import Simulator

from .mosaics_result import MosaicsResult

# from .template_iterator import BaseTemplateIterator
from .template_iterator import ResidueTemplateIterator


class MosaicsManager(BaseModel):
    """Class for importing, running, and exporting MOSAICS program data.

    TODO: Complete docstring

    Attributes
    ----------
    particle_stack : ParticleStack
        Stack of particle images with associated metadata (orientation, position,
        defocus) necessary for template matching.
    simulator : Simulator
        Instance of Simulator model from ttsim3d package. Holds the pdb file and
        associated atom positions, bfactors, etc. for simulating a 3D volume.
    template_iterator : TemplateIterator
        Iteration configuration model for describing how to iterate over the reference
        structure.
    """

    particle_stack: ParticleStack  # comes from Leopard-EM
    simulator: Simulator  # comes from ttsim3d
    template_iterator: ResidueTemplateIterator
    preprocessing_filters: PreprocessingFilters
    sim_removed_atoms_only: bool = False

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

        return cls(**data)

    def setup_image_stack(self) -> torch.Tensor:
        """Constructs the filtered image stack for the particle images.

        TODO: Complete docstring
        """
        # Extract the images and do pre-processing steps
        particle_images = self.particle_stack.construct_image_stack()
        particle_images_dft = torch.fft.rfftn(particle_images, dim=(-2, -1))
        particle_images_dft[..., 0, 0] = 0.0 + 0.0j  # Zero out DC component

        # Calculate and apply the filters for the particle image stack
        filter_stack = self.particle_stack.construct_filter_stack(
            self.preprocessing_filters, output_shape=particle_images_dft.shape[-2:]
        )
        particle_images_dft *= filter_stack

        # Normalize each particle image to mean zero variance 1
        squared_image_dft = torch.abs(particle_images_dft) ** 2
        squared_sum = torch.sum(squared_image_dft, dim=(-2, -1), keepdim=True)
        particle_images_dft /= torch.sqrt(squared_sum)

        # Normalize by the effective number of pixels in the particle images
        # (sum of the bandpass filter). See comments in 'match_template_manager.py'.
        bp_config = self.preprocessing_filters.bandpass_filter
        bp_filter_image = bp_config.calculate_bandpass_filter(
            particle_images_dft.shape[-2:]
        )
        dimensionality = bp_filter_image.sum()
        particle_images_dft *= dimensionality**0.5

        return particle_images_dft

    def setup_projection_filter_stack(self) -> torch.Tensor:
        """Constructs the filter stack for the projection images.

        TODO: Complete docstring
        """
        template_shape = self.particle_stack.original_template_size

        # Calculate the filters applied to each template (except for CTF)
        projective_filters = self.particle_stack.construct_filter_stack(
            self.preprocessing_filters,
            output_shape=(template_shape[-2], template_shape[-1] // 2 + 1),
        )

        # The best defocus values for each particle (+ astigmatism)
        defocus_u = self.particle_stack.absolute_defocus_u
        defocus_v = self.particle_stack.absolute_defocus_v
        defocus_angle = torch.tensor(self.particle_stack["astigmatism_angle"])

        # Keyword arguments for the CTF filter calculation call
        # NOTE: We currently enforce the parameters (other than the defocus values) are
        # all the same. This could be updated in the future...
        part_stk = self.particle_stack
        assert part_stk["pixel_size"].nunique() == 1
        assert part_stk["voltage"].nunique() == 1
        assert part_stk["spherical_aberration"].nunique() == 1
        assert part_stk["amplitude_contrast_ratio"].nunique() == 1
        assert part_stk["phase_shift"].nunique() == 1
        assert part_stk["ctf_B_factor"].nunique() == 1

        ctf_kwargs = {
            "voltage": part_stk["voltage"][0].item(),
            "spherical_aberration": part_stk["spherical_aberration"][0].item(),
            "amplitude_contrast": part_stk["amplitude_contrast_ratio"][0].item(),
            "b_factor": part_stk["ctf_B_factor"][0].item(),
            "phase_shift": part_stk["phase_shift"][0].item(),
            "pixel_size": part_stk["pixel_size"][0].item(),
            "image_shape": template_shape,
            "rfft": True,
            "fftshift": False,
        }

        # Calculate all the CTF filters for the particle images
        defocus = (defocus_u + defocus_v) / 2
        astigmatism = (defocus_u - defocus_v) / 2
        ctf_filters = calculate_ctf_2d(
            defocus=defocus * 1e-4,  # Angstrom to um
            astigmatism=astigmatism * 1e-4,  # Angstrom to um
            astigmatism_angle=defocus_angle,
            **ctf_kwargs,
        )

        return projective_filters * ctf_filters

    def run_mosaics(self, gpu_id: int) -> MosaicsResult:
        """Run the MOSAICS program.

        TODO: Complete docstring
        """
        if gpu_id == -1:
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{gpu_id}")

        particle_images_dft = self.setup_image_stack()
        projective_filters = self.setup_projection_filter_stack()

        # Orientations of all the particles
        euler_angles = torch.stack(
            (
                torch.tensor(self.particle_stack["refined_phi"]),
                torch.tensor(self.particle_stack["refined_theta"]),
                torch.tensor(self.particle_stack["refined_psi"]),
            ),
            dim=-1,
        )
        rot_mat = roma.euler_to_rotmat("ZYZ", euler_angles, degrees=True)
        rot_mat = rot_mat.float()

        # Pass tensors to device
        particle_images_dft = particle_images_dft.to(device)
        rot_mat = rot_mat.to(device)
        projective_filters = projective_filters.to(device)

        #####################################################
        ### Default (full-length) model cross-correlation ###
        #####################################################

        default_volume = self.simulator.run(gpu_ids=gpu_id)
        default_volume = torch.fft.fftshift(default_volume)
        default_volume_dft = torch.fft.rfftn(default_volume, dim=(-3, -2, -1))
        default_volume_dft = torch.fft.fftshift(default_volume_dft, dim=(-3, -2))

        default_cc = cross_correlate_particle_stack(
            particle_stack_dft=particle_images_dft,
            template_dft=default_volume_dft,
            rotation_matrices=rot_mat,
            projective_filters=projective_filters,
            mode="valid",
            batch_size=2048,
        )
        default_cc = torch.max(default_cc.view(default_cc.shape[0], -1), dim=-1).values

        ###################################################
        ### Iteration over alternate (truncated) models ###
        ###################################################

        # First pass to calculate the number of iterations
        atom_idx_iterator = self.template_iterator.atom_idx_iter()
        num_iters = sum(1 for _ in atom_idx_iterator)

        # NOTE: When the inverted flag is set to True, the iterator will return the
        # indices of the atoms that should NOT be removed. This is opposite of the
        # the 'sim_removed_atoms_only' flag.
        inverted = not self.sim_removed_atoms_only
        atom_idx_iterator = self.template_iterator.atom_idx_iter(inverted=inverted)
        chain_residue_iterator = self.template_iterator.chain_residue_iter()

        alternate_ccs = []
        alternate_chain_residue_pairs = []
        for chain_residue_pairs, atom_indices in tqdm.tqdm(
            zip(chain_residue_iterator, atom_idx_iterator),
            total=num_iters,
            desc="Iterating over alternate models",
        ):
            alternate_volume = self.simulator.run(
                gpu_ids=gpu_id, atom_indices=atom_indices
            )
            alternate_volume = torch.fft.fftshift(alternate_volume)
            alternate_volume_dft = torch.fft.rfftn(alternate_volume, dim=(-3, -2, -1))
            alternate_volume_dft = torch.fft.fftshift(
                alternate_volume_dft, dim=(-3, -2)
            )

            # Recalculate the cross-correlation with the alternate model
            # and take the maximum value over space
            alt_cc = cross_correlate_particle_stack(
                particle_stack_dft=particle_images_dft,
                template_dft=alternate_volume_dft,
                rotation_matrices=rot_mat,
                projective_filters=projective_filters,
                mode="valid",
                batch_size=2048,
            )
            alt_cc = torch.max(alt_cc.view(default_cc.shape[0], -1), dim=-1).values

            alternate_ccs.append(alt_cc)
            alternate_chain_residue_pairs.append(
                [(c, r) for c, r in zip(chain_residue_pairs[0], chain_residue_pairs[1])]
            )

        # Stack the alternate cross-correlation values into a single tensor
        alternate_ccs = torch.stack(alternate_ccs, dim=0)

        # Create the metadata for the alternate chain residues
        alternate_chain_residue_metadata = {
            f"alt_cc_{i}": chain_residue_pairs
            for i, chain_residue_pairs in enumerate(alternate_chain_residue_pairs)
        }

        return MosaicsResult(
            default_cross_correlation=default_cc.cpu().numpy(),
            alternate_cross_correlations=alternate_ccs.cpu().numpy(),  # type: ignore
            alternate_chain_residue_metadata=alternate_chain_residue_metadata,
            sim_removed_atoms_only=self.sim_removed_atoms_only,
        )
