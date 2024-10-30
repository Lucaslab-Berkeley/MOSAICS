import numpy as np
from typing import List, Tuple

from mosaics.data_structures.refine_template_result import RefineTemplateResult


class ParticleStack:
    """Class to store a stack of particle images and associated information about the
    particle locations and orientations. At minimum, a ParticleStack instance needs a
    box size (in pixels), a pixel size (in Angstroms), and an array holding a set of
    cropped particle images.

    Additional information per-particle can be provided, including the orientations,
    defocus parameters, z-scores, and MIP values.

    Attributes:
    -----------

        pixel_size (float): The size of the pixels in Angstroms.
        box_size (Tuple[int, int]): The size of the box in pixels.
        particle_images (np.ndarray): An array holding a set of cropped particle images.
        particle_locations (np.ndarray): Optional locations of the particles in pixels
            relative to the original micrograph. Default is None.
        particle_orientations (np.ndarray): Optional orientations of the particles in
            degrees (phi, theta, psi). Default is None.
        particle_defocus_parameters (np.ndarray): Optional defocus parameters of the
            particles in Angstroms (z1, z2, angle). Default is None.
        particle_z_scores (np.ndarray): Optional z-scores of the particles. Default is
            None.
        particle_mip_values (np.ndarray): Optional MIP values of the particles. Default is
            None.
        micrograph_reference_paths (List[str]): Optional list of paths to the micrograph
            reference files. Default is None.

    Methods:
    --------

    """

    pixel_size: float  # in Angstroms
    box_size: Tuple[int, int]  # in pixels
    particle_images: np.ndarray
    particle_locations: np.ndarray  # in pixels relative to original micrograph
    particle_orientations: np.ndarray  # in degrees (phi, theta, psi)
    particle_defocus_parameters: np.ndarray  # (z1, z2, angle) in Angstroms
    particle_z_scores: np.ndarray
    particle_mip_values: np.ndarray
    micrograph_reference_paths: List[str]

    @classmethod
    def from_refine_template_result(
        cls,
        refine_template_result: RefineTemplateResult,
        **kwargs,
    ):
        """Uses the held Micrograph object and refined template matching parameters to
        create a ParticleStack object. Additional keyword arguments are passed to the
        Micrograph.from_particle_stack method.
        """
        # Basic information on location and orientation
        box_size = refine_template_result.reference_template.box_size
        positions = refine_template_result.particle_locations
        orientations = np.stack(
            (
                refine_template_result.refined_phi,
                refine_template_result.refined_theta,
                refine_template_result.refined_psi,
            ),
            axis=-1,
        )
        # Calculate the defocus information per-particle in absolute units taking into
        # account the micrograph CTF fit
        ctf = refine_template_result.micrograph.ctf
        defocus_1 = ctf.defocus_1 + refine_template_result.refined_defocus[:, 0]
        defocus_2 = ctf.defocus_2 + refine_template_result.refined_defocus[:, 1]
        defocus_angle = ctf.defocus_angle + refine_template_result.refined_defocus[:, 2]
        defocus_parameters = np.stack(
            (defocus_1, defocus_2, defocus_angle),
            axis=-1,
        )

        # Information about the z-score and MIP values from 2DTM
        z_scores = refine_template_result.particle_z_scores
        mip_values = refine_template_result.refined_mip

        # Call the Micrograph.from_particle_stack method
        return refine_template_result.micrograph.from_particle_stack(
            box_size,
            positions,
            orientations,
            defocus_parameters,
            z_scores,
            mip_values,
            **kwargs,
        )

    @classmethod
    def concatenate(cls, particle_stacks):
        """Concatenate a list of ParticleStack instances."""

    def __init__(
        self,
        pixel_size: float,
        box_size: Tuple[int, int],
        particle_images: np.ndarray,
        particle_locations: np.ndarray = None,
        particle_orientations: np.ndarray = None,
        particle_defocus_parameters: np.ndarray = None,
        particle_z_scores: np.ndarray = None,
        particle_mip_values: np.ndarray = None,
        micrograph_reference_paths: List[str] = None,
    ):
        self.pixel_size = pixel_size
        self.box_size = box_size
        self.particle_images = particle_images
        self.particle_locations = particle_locations
        self.particle_orientations = particle_orientations
        self.particle_defocus_parameters = particle_defocus_parameters
        self.particle_z_scores = particle_z_scores
        self.particle_mip_values = particle_mip_values
        self.micrograph_reference_paths = micrograph_reference_paths

    def __add__(self, other):
        """Add two ParticleStack instances by combining the particle images and
        associated information.
        """
        assert (
            self.box_size == other.box_size
        ), "Cannot add ParticleStack objects with unequal box sizes"
        assert (
            self.pixel_size == other.pixel_size
        ), "Cannot add ParticleStack objects with unequal pixel sizes"

        new_particle_images = np.concatenate(
            [self.particle_images, other.particle_images], axis=0
        )

        # TODO: Check if any optional information is None and broadcast to new size
        new_particle_locations = np.concatenate(
            [self.particle_locations, other.particle_locations], axis=0
        )
        new_particle_orientations = np.concatenate(
            [self.particle_orientations, other.particle_orientations], axis=0
        )
        new_particle_defocus_parameters = np.concatenate(
            [self.particle_defocus_parameters, other.particle_defocus_parameters],
            axis=0,
        )
        new_particle_z_scores = np.concatenate(
            [self.particle_z_scores, other.particle_z_scores], axis=0
        )
        new_particle_mip_values = np.concatenate(
            [self.particle_mip_values, other.particle_mip_values], axis=0
        )
        new_micrograph_reference_paths = (
            self.micrograph_reference_paths + other.micrograph_reference_paths
        )

        return cls(
            self.pixel_size,
            self.box_size,
            new_particle_images,
            new_particle_locations,
            new_particle_orientations,
            new_particle_defocus_parameters,
            new_particle_z_scores,
            new_particle_mip_values,
            new_micrograph_reference_paths,
        )

    def to_json(self) -> dict:
        """Convert the ParticleStack object to a JSON-serializable dictionary."""
        return {
            "pixel_size": self.pixel_size,
            "particle_orientations": self.particle_orientations.tolist() if self.particle_orientations is not None else None,
            "particle_positions": self.particle_positions.tolist() if self.particle_positions is not None else None,
            "particle_defocus_parameters": self.particle_defocus_parameters.tolist() if self.particle_defocus_parameters is not None else None,
            # Note: particle_images_array is not included as it would be too large
        }

    @classmethod
    def from_json(cls, json_dict: dict) -> "ParticleStack":
        """Create a ParticleStack object from a JSON dictionary."""
        return cls(
            pixel_size=json_dict["pixel_size"],
            particle_images_array=None,  # This should be loaded separately
            particle_orientations=np.array(json_dict["particle_orientations"]) if json_dict["particle_orientations"] is not None else None,
            particle_positions=np.array(json_dict["particle_positions"]) if json_dict["particle_positions"] is not None else None,
            particle_defocus_parameters=np.array(json_dict["particle_defocus_parameters"]) if json_dict["particle_defocus_parameters"] is not None else None,
        )
