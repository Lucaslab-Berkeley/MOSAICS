"""Dataset classes for loading and caching input data for MOSAICS."""

ERR_MSG = (
    "Dataset has not been processed or loaded yet. "
    "Please either call process_inputs() or load_from_cache() first."
)


import numpy as np
import roma
import torch
from leopard_em.pydantic_models.config import PreprocessingFilters
from leopard_em.pydantic_models.data_structures import ParticleStack
from leopard_em.utils.backend_setup import setup_images_filters_particle_stack
from leopard_em.utils.ctf_utils import (
    _setup_ctf_kwargs_from_particle_stack,
    calculate_ctf_filter_stack_full_args,
)
from pydantic import BaseModel
from ttsim3d.models import Simulator


class MosaicsDataset(BaseModel):
    """Dataset class for processing, loading, and caching inputs for MOSAICS.

    Attributes
    ----------
    particle_stack : ParticleStack
        The particle stack containing the particle images and metadata.
    preprocessing_filters : PreprocessingFilters
        The pre-processing filters to apply to the particle images and templates.

    Properties
    ----------
    rotation_matrices : torch.Tensor
        The per-particle rotation matrices derived from the particle stack Euler angles.
        Shape of (N, 3, 3).
    projective_filters : torch.Tensor
        The per-particle projective filters derived from the whitening filter, any other
        curve filters, the contrast transfer function, etc. Shape of (N, h, w) where h
        and w correspond the the heigh/width of a template projection.
    particle_images : torch.Tensor
        The pre-processed particle images. Shape of (N, H, W) where H and W correspond
        to the height/width of the extracted particle images.

    Methods
    -------
    process_inputs(device: torch.device, simulator: Simulator) -> None
        Process the raw inputs (particle stack and template shape) into the necessary
        variables for running MOSAICS.
    save_to_cache(cache_path: str) -> None
        Saves the pre-processed dataset to a cache file for future reuse.
    load_from_cache(cache_path: str, device: torch.device) -> None
        Load in a pre-processed dataset from a cache file.
    """

    particle_stack: ParticleStack
    preprocessing_filters: PreprocessingFilters

    _rotation_matrices: torch.Tensor | None = None
    _projective_filters: torch.Tensor | None = None
    _particle_images: torch.Tensor | None = None

    @property
    def rotation_matrices(self) -> torch.Tensor:
        if self._rotation_matrices is None:
            raise ValueError(ERR_MSG)
        return self._rotation_matrices

    @property
    def projective_filters(self) -> torch.Tensor:
        if self._projective_filters is None:
            raise ValueError(ERR_MSG)
        return self._projective_filters

    @property
    def particle_images(self) -> torch.Tensor:
        if self._particle_images is None:
            raise ValueError(ERR_MSG)
        return self._particle_images

    def process_inputs(self, device: torch.device, simulator: Simulator) -> None:
        """Process inputs to necessary variables for running MOSAICS.

        Parameters
        ----------
        device : torch.device
            The device to process the dataset on and load tensors onto.
        simulator : Simulator
            Simulator object defining the template; only its volume shape (the
            template projection size) is used here.

        Returns
        -------
        None
        """
        template_h, template_w = simulator.volume_shape[-2:]

        # NOTE: Leopard-EM only uses the template for its device, its shape (to size
        # the projective filters), and a template DFT which MOSAICS does not need
        # since projections are computed from atoms. Pass a lightweight placeholder.
        template_placeholder = torch.zeros((1, template_h, template_w), device=device)
        particle_images_dft, _, projective_filters = (
            setup_images_filters_particle_stack(
                particle_stack=self.particle_stack,
                preprocessing_filters=self.preprocessing_filters,
                template=template_placeholder,
            )
        )
        particle_images = torch.fft.irfftn(particle_images_dft, dim=(-2, -1))
        particle_images *= torch.numel(particle_images[0]) ** 0.5  # normalize

        # Calculate the per-particle CTF array and combine it with projective filters
        defocus_u, defocus_v = self.particle_stack.get_absolute_defocus()
        defocus_angle = torch.tensor(self.particle_stack["astigmatism_angle"])
        ctf_kwargs = _setup_ctf_kwargs_from_particle_stack(
            self.particle_stack, (template_h, template_w)
        )
        ctf_filters = calculate_ctf_filter_stack_full_args(
            defocus_u=defocus_u,  # in Angstrom
            defocus_v=defocus_v,  # in Angstrom
            astigmatism_angle=defocus_angle,  # in degrees
            defocus_offsets=torch.Tensor([0.0]),
            pixel_size_offsets=torch.Tensor([0.0]),
            **ctf_kwargs,
        )

        # NOTE: Leopard-EM returns the projective filters on the template's device,
        # while the CTF filters are computed on the CPU.
        ctf_filters = ctf_filters.squeeze(0).to(projective_filters.device)
        projective_filters = projective_filters * ctf_filters

        # Grab the per-particle orientations preferring refined angles (if they exist)
        euler_angles = self.particle_stack.get_euler_angles(prefer_refined_angles=True)
        rotation_matrices = roma.euler_to_rotmat("ZYZ", euler_angles, degrees=True)
        rotation_matrices = rotation_matrices.float()

        # Store the processed variables for later use
        self._rotation_matrices = rotation_matrices.to(device)
        self._projective_filters = projective_filters.to(device)
        self._particle_images = particle_images.to(device)

    def save_to_cache(self, cache_path: str) -> None:
        """Saves the pre-processed dataset to a cache file for future reuse.

        Parameters
        ----------
        cache_path : str
            The file path where the cached dataset will be saved. This should be a numpy
            .npz file.

        Returns
        -------
        None
        """
        rotation_matrices = self._rotation_matrices.cpu().numpy()
        projective_filters = self._projective_filters.cpu().numpy()
        particle_images = self._particle_images.cpu().numpy()

        np.savez(
            cache_path,
            rotation_matrices=rotation_matrices,
            projective_filters=projective_filters,
            particle_images=particle_images,
        )

    def load_from_cache(self, cache_path: str, device: torch.device) -> None:
        """Load in a pre-processed dataset from a cache file.

        NOTE: This assumes that there have been no changes to the particle stack or
        pre-processing filters (which are owned by this class) OR the template structure
        size defined by the Simulator (owned by the MosaicsManager). Any changes to
        these program inputs would require re-processing and re-saving the dataset.
        Caches written by older versions (with template volumes) can still be loaded;
        the volumes are ignored.

        Parameters
        ----------
        cache_path : str
            The file path where the cached dataset is stored. This should be a numpy
            .npz file.
        device : torch.device
            The device to load the cached dataset onto.

        Returns
        -------
        None
        """
        cache_data = np.load(cache_path)

        # Ensure the required keys are present in the cache file
        required_keys = [
            "rotation_matrices",
            "projective_filters",
            "particle_images",
        ]
        for key in required_keys:
            if key not in cache_data:
                raise ValueError(f"Cache file is missing required key: {key}")

        # fmt: off
        self._rotation_matrices     = torch.from_numpy(cache_data["rotation_matrices"]).to(device)
        self._projective_filters    = torch.from_numpy(cache_data["projective_filters"]).to(device)
        self._particle_images       = torch.from_numpy(cache_data["particle_images"]).to(device)
        # fmt: on
