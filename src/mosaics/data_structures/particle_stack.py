from typing import List
from typing import Optional

import numpy as np

from mosaics.data_structures.micrograph import Micrograph
from mosaics.utils import parse_out_coordinates_result


class ParticleStack:
    """Class to store a stack of particle images and associated information
    about the particle locations and orientations. At minimum, a ParticleStack
    instance needs a box size (in pixels), a pixel size (in Angstroms), and an
    array holding a set of cropped particle images.

    Additional information per-particle can be provided, including the
    orientations, defocus parameters, z-scores, and MIP values.

    Attributes:
    -----------

        pixel_size (float): The size of the pixels in Angstroms.
        box_size (Tuple[int, int]): The size of the box in pixels.
        particle_images (np.ndarray): An array holding a set of cropped
            particle images.
        particle_positions (np.ndarray): Optional locations of the particles
            in pixels relative to the original micrograph. Default is None.
        particle_orientations (np.ndarray): Optional orientations of the
            particles in degrees (phi, theta, psi). Default is None.
        particle_defocus_parameters (np.ndarray): Optional defocus parameters
            of the particles in Angstroms (z1, z2, angle). Default is None.
        particle_z_scores (np.ndarray): Optional z-scores of the particles.
            Default is None.
        particle_mip_values (np.ndarray): Optional MIP values of the particles.
            Default is None.
        micrograph_reference_paths (List[str]): Optional list of paths to the
        micrograph reference files. Default is None.

    Methods:
    --------

    """

    # TODO: Sort out optional attributes and default values
    pixel_size: float  # in Angstroms
    box_size: tuple[int, int]  # in pixels
    particle_images: np.ndarray
    particle_positions: np.ndarray  # in pixels relative to original micrograph
    particle_orientations: np.ndarray  # in degrees (phi, theta, psi)
    particle_defocus_parameters: np.ndarray  # (z1, z2, angle) in Angstroms
    particle_z_scores: np.ndarray
    particle_mip_values: np.ndarray
    micrograph_reference_paths: Optional[List[str]]

    @classmethod
    def from_out_coordinates_and_micrograph(
        cls,
        out_coordinates_path: str,
        micrograph: "Micrograph",
        box_size: tuple[int, int],
        **kwargs,
    ):
        """Create a RefineTemplateResult object from an out_coordinates.txt
        file (produced by cisTEM make_template_result program after
        refine_template) and a Micrograph object as a reference to the original
        image.

        TODO: complete docstring
        """
        coord_df = parse_out_coordinates_result(out_coordinates_path)

        # Get pixel coordinates for each of the particles
        positions_x = coord_df["X"].values
        positions_y = coord_df["Y"].values
        positions_x = np.round(positions_x / micrograph.pixel_size).astype(int)
        positions_y = np.round(positions_y / micrograph.pixel_size).astype(int)

        # Z-scores (SNR) and orientations per particle
        particle_z_scores = coord_df["Peak"].values
        particle_orientations = np.stack(
            (
                coord_df["Psi"].values,
                coord_df["Theta"].values,
                coord_df["Phi"].values,
            ),
            axis=-1,
        )

        # Calculate the absolute defocus parameters for each particle
        if micrograph.ctf is not None:
            ctf = micrograph.ctf
            defocus_1 = ctf.defocus_1 + coord_df["Z"].values
            defocus_2 = ctf.defocus_2 + coord_df["Z"].values
            defocus_angle = np.full(defocus_1.size, ctf.astigmatism_azimuth)
            particle_defocus_parameters = np.stack(
                (defocus_1, defocus_2, defocus_angle),
                axis=-1,
            )
        else:
            particle_defocus_parameters = None

        return micrograph.to_particle_stack(
            box_size=box_size,
            positions_x=positions_x,
            positions_y=positions_y,
            particle_orientations=particle_orientations,
            particle_defocus_parameters=particle_defocus_parameters,
            particle_z_scores=particle_z_scores,
            **kwargs,
        )

    @classmethod
    def concatenate(cls, particle_stacks):
        """Concatenate a list of ParticleStack instances."""
        new_particle_stack = particle_stacks[0]
        for particle_stack in particle_stacks[1:]:
            new_particle_stack += particle_stack

        return new_particle_stack

    def __init__(
        self,
        pixel_size: float,
        box_size: tuple[int, int],
        particle_images: np.ndarray,
        particle_positions: np.ndarray = None,
        particle_orientations: np.ndarray = None,
        particle_defocus_parameters: np.ndarray = None,
        particle_z_scores: np.ndarray = None,
        particle_mip_values: np.ndarray = None,
        micrograph_reference_paths: Optional[List[str]] = None,
    ):
        self.pixel_size = pixel_size
        self.box_size = box_size
        self.particle_images = particle_images
        self.particle_positions = particle_positions
        self.particle_orientations = particle_orientations
        self.particle_defocus_parameters = particle_defocus_parameters
        self.particle_z_scores = particle_z_scores
        self.particle_mip_values = particle_mip_values
        self.micrograph_reference_paths = micrograph_reference_paths

    def __repr__(self):
        raise NotImplementedError

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

        # TODO: Check if any optional information is None and broadcast to new
        # size
        new_particle_positions = np.concatenate(
            [self.particle_positions, other.particle_positions], axis=0
        )
        new_particle_orientations = np.concatenate(
            [self.particle_orientations, other.particle_orientations], axis=0
        )
        new_particle_defocus_parameters = np.concatenate(
            [
                self.particle_defocus_parameters,
                other.particle_defocus_parameters,
            ],
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

        return ParticleStack(
            self.pixel_size,
            self.box_size,
            new_particle_images,
            new_particle_positions,
            new_particle_orientations,
            new_particle_defocus_parameters,
            new_particle_z_scores,
            new_particle_mip_values,
            new_micrograph_reference_paths,
        )

    # def to_json(self) -> dict:
    #     """Convert the ParticleStack object to a JSON-serializable dictionary
    #     and return it.
    #     """
    #     # TODO: Export the particle stack array as a separate .mrc file

    #     return {
    #         "pixel_size": self.pixel_size,
    #         "particle_orientations": (
    #             self.particle_orientations.tolist()
    #             if self.particle_orientations is not None
    #             else None
    #         ),
    #         "particle_positions": (
    #             self.particle_positions.tolist()
    #             if self.particle_positions is not None
    #             else None
    #         ),
    #         "particle_defocus_parameters": (
    #             self.particle_defocus_parameters.tolist()
    #             if self.particle_defocus_parameters is not None
    #             else None
    #         ),
    #     }

    # @classmethod
    # def from_json(cls, json_dict: dict) -> "ParticleStack":
    #     """Create a ParticleStack object from a JSON dictionary."""
    #     return cls(
    #         pixel_size=json_dict["pixel_size"],
    #         particle_images=None,  # This should be loaded separately
    #         particle_orientations=(
    #             np.array(json_dict["particle_orientations"])
    #             if json_dict["particle_orientations"] is not None
    #             else None
    #         ),
    #         particle_positions=(
    #             np.array(json_dict["particle_positions"])
    #             if json_dict["particle_positions"] is not None
    #             else None
    #         ),
    #         particle_defocus_parameters=(
    #             np.array(json_dict["particle_defocus_parameters"])
    #             if json_dict["particle_defocus_parameters"] is not None
    #             else None
    #         ),
    #     )

    # TODO: to/from .star
