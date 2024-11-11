from typing import Optional

import mrcfile
import numpy as np
import pandas as pd
import starfile

from mosaics.data_structures.micrograph import Micrograph
from mosaics.utils import parse_out_coordinates_result

STAR_COLUMNS = [
    "ParticleIndex",
    "ImageStackPath",
    "ParticleClass",
    "PixelCoordinateX",
    "PixelCoordinateY",
    "OrientationPhi",
    "OrientationTheta",
    "OrientationPsi",
    "Defocus1",
    "Defocus2",
    "DefocusAngle",
    "MicrographPath",
    "MicrographPSDPath",
]


class ParticleStack:
    """Class to store a stack of particle images and associated information
    about the particle locations and orientations. At minimum, a ParticleStack
    instance needs a box size (in pixels), a pixel size (in Angstroms), and an
    array holding a set of cropped particle images.

    Additional information per-particle is also stored and used when running
    MOSAICS. This includes the the orientation of the particles (from 2DTM),
    the best defocus parameters for each particle (from 2DTM), and the power
    spectral density of the original micrograph (referenced by file path).

    There are two main workflows for instantiating a ParticleStack object, each
    of which are detailed in the examples gallery, but in summary:
        1. From a set of micrographs and associated out_coordinates.txt files
            (produced by cisTEM refine_template) using the class method
            `from_out_coordinates_and_micrograph`.
        2. From a pre-existing stack of particle images (.mrcs file(s)) and a
            custom STAR file defining necessary metadata using the class
            method `from_star`.

    Attributes:
    -----------

        particle_images (np.ndarray): An array holding a set of cropped
            particle images. Shape (N, H, W) where N is the number of particles
            and H, W are the height and width of the particle images.

        pixel_size (float): The size of the pixels in Angstroms.
        box_size (Tuple[int, int]): The size of the box in pixels.
        spherical_aberration (float): The spherical aberration of the optics
            used to collect the images of the particles in mm. Default is 2.7
            mm.
        voltage (float): The voltage of the microscope in kV. Default is 300
            kV.
        amplitude_contrast (float): The amplitude contrast for the optics used
            to collect images of the particles. Default is 0.07.
        B_factor (float): The B-factor of the particles in Angstroms^2. Default
            is 0.0.

        particle_index (np.ndarray): Index of the particle in the reference
            stack (corresponds to particle_image_stack_paths). Default is None.
        particle_class (np.ndarray): Optional array describing class labels for
            each particle. Default is None.
        particle_positions (np.ndarray): Optional locations of the particles
            in pixels relative to the original micrograph. Shape of (N, 2)
            with N being the number of particles and the columns being (x, y).
            Default is None.
        particle_orientations (np.ndarray): Optional orientations of the
            particles in degrees using the ZYZ convention (phi, theta, psi).
            Shape of (N, 3) with N being the number of particles and the
            columns being the three angles. Default is None.
        particle_defocus_parameters (np.ndarray): Optional defocus parameters
            of the particles. Shape of (N, 3) where N is the number of
            particles. The first column is the major defocus and the second
            column being the minor defocus, both in units of Angstroms, and the
            third column is the defocus angle, in degrees (z1, z2, angle).
            Default is None.
        particle_z_scores (np.ndarray): Optional z-scores of the particles.
            Default is None.
        particle_mip_values (np.ndarray): Optional MIP values of the particles.
            Default is None.
        particle_image_stack_paths (list[str]): List of paths to the particle
            stack .mrcs files. Default is None.
        particle_micrograph_paths (list[str]): List of paths to the original
            micrograph .mrc files. Default is None.
        particle_psd_paths (list[str]): List of paths to the micrograph
            power spectral density files. Default is None.

    Methods:
    --------

    """

    # ParticleStack scalar metadata
    pixel_size: float  # in Angstroms
    box_size: tuple[int, int]  # in pixels

    # Image stack data (from Micrograph or loaded from .mrcs)
    particle_images: np.ndarray

    # Tabular per-particle metadata
    particle_index: np.ndarray
    particle_class: np.ndarray
    particle_positions: np.ndarray  # in pixels relative to original micrograph
    particle_orientations: np.ndarray  # in degrees (phi, theta, psi)
    particle_defocus_parameters: np.ndarray  # (z1, z2, angle) in Angstroms
    particle_z_scores: np.ndarray
    particle_mip_values: np.ndarray

    # Reference to image stack, micrograph, and psd files
    particle_image_stack_paths: Optional[list[str]]
    particle_micrograph_paths: Optional[list[str]]
    particle_psd_paths: Optional[list[str]]

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

        TODO: Figure out way to handle pre-defined PSD classes rather than
        just paths to the PSD files.
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
        if micrograph.contrast_transfer_function is not None:
            ctf = micrograph.contrast_transfer_function
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
    def from_star(cls, star_path: str):
        """Create a ParticleStack object from a STAR file.

        TODO: complete docstring

        """
        star_dict = starfile.read(star_path, always_dict=True)
        attr_df = star_dict["particle_stack_attributes"]
        table_df = star_dict["particle_stack_table"]

        # # TODO: Further data frame verification
        # assert set(table_df.columns) == set(STAR_COLUMNS), (
        #     "The columns in the STAR file dont match the expected columns. "
        #     f"Expected: {STAR_COLUMNS}, Found: {table_df.columns}"
        # )

        # Pixel size and box size
        pixel_size = attr_df["PixelSize"].values[0]
        box_size = (attr_df["BoxSize"].values[0], attr_df["BoxSize"].values[0])
        voltage = attr_df["Voltage"].values[0]
        spherical_aberration = attr_df["SphericalAberration"].values[0]
        amplitude_contrast_ratio = attr_df["AmplitudeContrastRatio"].values[0]
        B_factor = attr_df["BFactor"].values[0]

        # Tabular data per-particle
        # fmt: off
        particle_index = table_df["ParticleIndex"].values
        particle_class = table_df["ParticleClass"].values
        particle_positions = table_df[["PixelCoordinateX", "PixelCoordinateY"]].values  # noqa: E501
        particle_orientations = table_df[["OrientationPhi", "OrientationTheta", "OrientationPsi"]].values  # noqa: E501
        particle_defocus_parameters = table_df[["Defocus1", "Defocus2", "DefocusAngle"]].values  # noqa: E501
        particle_image_stack_paths = table_df["ImageStackPath"].values
        particle_micrograph_paths = table_df["MicrographPath"].values
        particle_psd_paths = table_df["MicrographPSDPath"].values
        # fmt: on

        # Load in the particle images from the stack paths
        last_image_stack_path = None
        to_load_paths = []
        for i, image_stack_path in enumerate(particle_image_stack_paths):
            if image_stack_path != last_image_stack_path:
                to_load_paths.append(image_stack_path)
                last_image_stack_path = image_stack_path
        particle_images = np.concatenate(
            [mrcfile.open(path).data for path in to_load_paths], axis=0
        )

        return cls(
            pixel_size=pixel_size,
            box_size=box_size,
            voltage=voltage,
            spherical_aberration=spherical_aberration,
            amplitude_contrast_ratio=amplitude_contrast_ratio,
            B_factor=B_factor,
            particle_images=particle_images,
            particle_index=particle_index,
            particle_class=particle_class,
            particle_positions=particle_positions,
            particle_orientations=particle_orientations,
            particle_defocus_parameters=particle_defocus_parameters,
            particle_image_stack_paths=particle_image_stack_paths,
            particle_micrograph_paths=particle_micrograph_paths,
            particle_psd_paths=particle_psd_paths,
        )

    def to_star(self, star_path: str, mrcs_path: str):
        """Write the ParticleStack object to a STAR file.

        TODO: complete docstring
        """
        # Export the particle stack array as a separate .mrcs file
        with mrcfile.new(mrcs_path, overwrite=True) as mrc:
            mrc.set_data(self.particle_images)
            mrc.voxel_size = self.pixel_size  # Is this correct?

        # Create the STAR file
        df_params = pd.DataFrame(
            {
                "PixelSize": [self.pixel_size],
                "BoxSize": [self.box_size[0]],
                "Voltage": [self.voltage],
                "SphericalAberration": [self.spherical_aberration],
                "AmplitudeContrastRatio": [self.amplitude_contrast_ratio],
                "BFactor": [self.B_factor],
            }
        )

        df_particles = pd.DataFrame(
            {
                "ParticleIndex": self.particle_index,
                "ImageStackPath": self.particle_image_stack_paths,
                "ParticleClass": self.particle_class,
                "PixelCoordinateX": self.particle_positions[:, 0],
                "PixelCoordinateY": self.particle_positions[:, 1],
                "OrientationPhi": self.particle_orientations[:, 0],
                "OrientationTheta": self.particle_orientations[:, 1],
                "OrientationPsi": self.particle_orientations[:, 2],
                "Defocus1": self.particle_defocus_parameters[:, 0],
                "Defocus2": self.particle_defocus_parameters[:, 1],
                "DefocusAngle": self.particle_defocus_parameters[:, 2],
                "MicrographPath": self.particle_micrograph_paths,
                "MicrographPSDPath": self.particle_psd_paths,
            }
        )

        star_dict = {
            "particle_stack_attributes": df_params,
            "particle_stack_table": df_particles,
        }

        starfile.write(star_path, star_dict)

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
        voltage: float = 300.0,
        spherical_aberration: float = 2.7,
        amplitude_contrast_ratio: float = 0.07,
        B_factor: float = 0.0,
        particle_index: np.ndarray = None,
        particle_class: np.ndarray = None,
        particle_positions: np.ndarray = None,
        particle_orientations: np.ndarray = None,
        particle_defocus_parameters: np.ndarray = None,
        particle_image_stack_paths: np.ndarray = None,
        particle_micrograph_paths: np.ndarray = None,
        particle_psd_paths: np.ndarray = None,
    ):
        self.pixel_size = pixel_size
        self.box_size = box_size
        self.voltage = voltage
        self.spherical_aberration = spherical_aberration
        self.amplitude_contrast_ratio = amplitude_contrast_ratio
        self.B_factor = B_factor
        self.particle_images = particle_images
        self.particle_index = particle_index
        self.particle_class = particle_class
        self.particle_positions = particle_positions
        self.particle_orientations = particle_orientations
        self.particle_defocus_parameters = particle_defocus_parameters
        self.particle_image_stack_paths = particle_image_stack_paths
        self.particle_micrograph_paths = particle_micrograph_paths
        self.particle_psd_paths = particle_psd_paths

    def __repr__(self) -> str:
        """Get string representation of the ParticleStack object."""
        mem_location = hex(id(self))
        string = f"ParticleStack object at {mem_location}:"
        string += f"\n\t{self.particle_images.shape[0]} particles."
        string += f"\n\t{self.box_size} pixels at {self.pixel_size} Å/pixel."
        string += f"""\n\tIncludes positions: {
            self.particle_positions is not None
        }"""
        string += f"""\n\tIncludes orientations: {
            self.particle_orientations is not None
        }"""
        string += f"""\n\tIncludes defocus parameters: {
            self.particle_defocus_parameters is not None
        }"""

        return string

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
