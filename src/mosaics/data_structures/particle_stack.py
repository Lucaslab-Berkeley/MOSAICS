# import mrcfile
import os
import typing
from typing import Literal
from typing import Optional

import numpy as np
import pandas as pd
import starfile

from mosaics.data_structures.micrograph import Micrograph
from mosaics.utils import get_cropped_region_of_image
from mosaics.utils import parse_out_coordinates_result


class OpticsGroups:
    """Class to store information related to microscope optics and image
    processing parameters necessary for constructing filters for each member
    of a ParticleStack in MOSAICS. An instance of the OpticsGroups class is
    indexable by both the optics group name and number which returns a
    dictionary for the corresponding optics group.

    Attributes:
    -----------

        optics_group_name (list[str]): List of names for each optics group.
        optics_group_number (list[int]): List of unique integers for each
            optics group.
        micrograph_original_pixel_size (list[float]): List of pixel sizes for
            the micrographs, in Angstroms.
        micrograph_original_shape_x (list[int]): List of the original size
            of the micrographs in the x-dimension.
        micrograph_original_shape_y (list[int]): List of the original size
            of the micrographs in the y-dimension.
        micrograph_psd_reference (list[str]): List of paths to pre-calculated
            power spectral density (PSD) files for the micrographs.
        voltage (list[float]): List of the voltage of the microscope in kV.
        spherical_aberration (list[float]): List of the spherical aberration
            of the microscope optics in mm.
        amplitude_contrast_ratio (list[float]): List of the amplitude contrast
            of the microscope optics.
        additional_phase_shift (list[float]): List of additional phase shifts
            applied to the images.
        defocus_1 (list[float]): Major defocus value for images (in Angstroms).
        defocus_2 (list[float]): Minor defocus value for images (in Angstroms).
        astigmatism_azimuth (list[float]): Defocus Astigmatism angle for images
        image_pixel_size (list[float]): List of pixel sizes for the images
            stacks, in Angstroms.
        image_shape_x (list[int]): List of the size of the image stacks in
            the x-dimension.
        image_shape_y (list[int]): List of the size of the image stacks in
            the y-dimension.
        image_shape_z (list[int]): List of the size of the image stacks in
            the z-dimension (number of particles).
        image_dimensionality (list[int]): List of the dimensionality of the
            image stacks. Usually 3.

    """

    optics_group_name: list[str]
    optics_group_number: list[int]
    micrograph_original_pixel_size: list[float]
    micrograph_original_shape_x: list[int]
    micrograph_original_shape_y: list[int]
    micrograph_psd_reference: list[str]
    voltage: list[float]
    spherical_aberration: list[float]
    amplitude_contrast_ratio: list[float]
    additional_phase_shift: list[float]
    defocus_1: list[float]
    defocus_2: list[float]
    astigmatism_azimuth: list[float]
    image_pixel_size: list[float]
    image_shape_x: list[int]
    image_shape_y: list[int]
    image_shape_z: list[int]
    image_dimensionality: list[int]

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        """Instantiate a new OpticsGroups object from a DataFrame, likely
        parsed from a star file.

        """
        optics_group_name = df["mosaicsOpticsGroupName"].values
        optics_group_number = df["mosaicsOpticsGroupNumber"].values
        micrograph_original_pixel_size = df[
            "mosaicsMicrographOriginalPixelSize"
        ].values
        micrograph_original_shape_x = df[
            "mosaicsMicrographOriginalShapeX"
        ].values
        micrograph_original_shape_y = df[
            "mosaicsMicrographOriginalShapeY"
        ].values
        micrograph_psd_reference = df["mosaicsMicrographPSDReference"].values
        voltage = df["mosaicsVoltage"].values
        spherical_aberration = df["mosaicsSphericalAberration"].values
        amplitude_contrast_ratio = df["mosaicsAmplitudeContrastRatio"].values
        additional_phase_shift = df["mosaicsAdditionalPhaseShift"].values
        defocus_1 = df["mosaicsDefocus1"].values
        defocus_2 = df["mosaicsDefocus2"].values
        astigmatism_azimuth = df["mosaicsAstigmatismAzimuth"].values
        image_pixel_size = df["mosaicsImagePixelSize"].values
        image_shape_x = df["mosaicsImageShapeX"].values
        image_shape_y = df["mosaicsImageShapeY"].values
        image_shape_z = df["mosaicsImageShapeZ"].values
        image_dimensionality = df["mosaicsImageDimensionality"].values

        return cls(
            optics_group_name=optics_group_name,
            optics_group_number=optics_group_number,
            micrograph_original_pixel_size=micrograph_original_pixel_size,
            micrograph_original_shape_x=micrograph_original_shape_x,
            micrograph_original_shape_y=micrograph_original_shape_y,
            micrograph_psd_reference=micrograph_psd_reference,
            voltage=voltage,
            spherical_aberration=spherical_aberration,
            amplitude_contrast_ratio=amplitude_contrast_ratio,
            additional_phase_shift=additional_phase_shift,
            defocus_1=defocus_1,
            defocus_2=defocus_2,
            astigmatism_azimuth=astigmatism_azimuth,
            image_pixel_size=image_pixel_size,
            image_shape_x=image_shape_x,
            image_shape_y=image_shape_y,
            image_shape_z=image_shape_z,
            image_dimensionality=image_dimensionality,
        )

    def to_df(self) -> pd.DataFrame:
        """Export the OpticsGroups object to a DataFrame."""
        # fmt: off
        df = pd.DataFrame()
        df["mosaicsOpticsGroupName"] = self.optics_group_name
        df["mosaicsOpticsGroupNumber"] = self.optics_group_number
        df["mosaicsMicrographOriginalPixelSize"] = (
            self.micrograph_original_pixel_size
        )
        df["mosaicsMicrographOriginalShapeX"] = (
            self.micrograph_original_shape_x
        )
        df["mosaicsMicrographOriginalShapeY"] = (
            self.micrograph_original_shape_y
        )
        df["mosaicsMicrographPSDReference"] = self.micrograph_psd_reference
        df["mosaicsVoltage"] = self.voltage
        df["mosaicsSphericalAberration"] = self.spherical_aberration
        df["mosaicsAmplitudeContrastRatio"] = self.amplitude_contrast_ratio
        df["mosaicsAdditionalPhaseShift"] = self.additional_phase_shift
        df["mosaicsDefocus1"] = self.defocus_1
        df["mosaicsDefocus2"] = self.defocus_2
        df["mosaicsAstigmatismAzimuth"] = self.astigmatism_azimuth
        df["mosaicsImagePixelSize"] = self.image_pixel_size
        df["mosaicsImageShapeX"] = self.image_shape_x
        df["mosaicsImageShapeY"] = self.image_shape_y
        df["mosaicsImageShapeZ"] = self.image_shape_z
        df["mosaicsImageDimensionality"] = self.image_dimensionality

        return df

    @classmethod
    def from_micrograph(
        cls,
        micrograph: "Micrograph",
        box_size: tuple[int, int],
        num_particles: int,
        group_name: str = None,
        group_number: int = 0,
    ):
        """Create a singular OpticsGroups object from a Micrograph object."""
        if micrograph.contrast_transfer_function is None:
            raise ValueError(
                "Micrograph object does not have an associated CTF."
            )

        # TODO: Other assertions to ensure the micrograph object is valid

        # Set default group name to micrograph file base name
        if group_name is None and micrograph.image_path is not None:
            group_name = os.path.basename(micrograph.image_path)

        pixel_size = micrograph.pixel_size
        # TODO: Check if this is correct
        shape_x = micrograph.image_array.shape[1]
        shape_y = micrograph.image_array.shape[0]
        # TODO: Figure out how to get the psd reference file path intelligently
        # psd_reference = micrograph.power_spectral_density

        _ctf = micrograph.contrast_transfer_function
        voltage = _ctf.voltage
        spherical_aberration = _ctf.spherical_aberration
        amplitude_contrast_ratio = _ctf.amplitude_contrast_ratio
        additional_phase_shift = _ctf.additional_phase_shift
        defocus_1 = _ctf.defocus_1
        defocus_2 = _ctf.defocus_2
        astigmatism_azimuth = _ctf.astigmatism_azimuth

        # NOTE: Currently assumed that pixel size is same with the micrograph
        image_pixel_size = pixel_size
        image_shape_x = box_size[1]
        image_shape_y = box_size[0]
        image_shape_z = num_particles
        image_dimensionality = 3

        return cls(
            optics_group_name=[group_name],
            optics_group_number=[group_number],
            micrograph_original_pixel_size=[pixel_size],
            micrograph_original_shape_x=[shape_x],
            micrograph_original_shape_y=[shape_y],
            micrograph_psd_reference=[None],
            voltage=[voltage],
            spherical_aberration=[spherical_aberration],
            amplitude_contrast_ratio=[amplitude_contrast_ratio],
            additional_phase_shift=[additional_phase_shift],
            defocus_1=[defocus_1],
            defocus_2=[defocus_2],
            astigmatism_azimuth=[astigmatism_azimuth],
            image_pixel_size=[image_pixel_size],
            image_shape_x=[image_shape_x],
            image_shape_y=[image_shape_y],
            image_shape_z=[image_shape_z],
            image_dimensionality=[image_dimensionality],
        )

    def __init__(
        self,
        optics_group_name,
        optics_group_number,
        micrograph_original_pixel_size,
        micrograph_original_shape_x,
        micrograph_original_shape_y,
        micrograph_psd_reference,
        voltage,
        spherical_aberration,
        amplitude_contrast_ratio,
        additional_phase_shift,
        defocus_1,
        defocus_2,
        astigmatism_azimuth,
        image_pixel_size,
        image_shape_x,
        image_shape_y,
        image_shape_z,
        image_dimensionality,
    ):
        self.optics_group_name = optics_group_name
        self.optics_group_number = optics_group_number
        self.micrograph_original_pixel_size = micrograph_original_pixel_size
        self.micrograph_original_shape_x = micrograph_original_shape_x
        self.micrograph_original_shape_y = micrograph_original_shape_y
        self.micrograph_psd_reference = micrograph_psd_reference
        self.voltage = voltage
        self.spherical_aberration = spherical_aberration
        self.amplitude_contrast_ratio = amplitude_contrast_ratio
        self.additional_phase_shift = additional_phase_shift
        self.defocus_1 = defocus_1
        self.defocus_2 = defocus_2
        self.astigmatism_azimuth = astigmatism_azimuth
        self.image_pixel_size = image_pixel_size
        self.image_shape_x = image_shape_x
        self.image_shape_y = image_shape_y
        self.image_shape_z = image_shape_z
        self.image_dimensionality = image_dimensionality

    def __getitem__(self, key: str | int) -> dict:
        """For indexing the OpticsGroups object by optics group name or number.
        Does assertion check to ensure the key is valid.
        """
        _in_group_name = key in self.optics_group_name
        _in_group_number = key in self.optics_group_number
        if not _in_group_name and not _in_group_number:
            raise ValueError(
                f"Key {key} absent from optics group names or numbers."
            )

        # Convert string-based key to integer index
        if isinstance(key, str):
            idx = self.optics_group_name.index(key)
        elif isinstance(key, int):
            idx = self.optics_group_number.index(key)
        else:
            raise ValueError("Key must be either a string or an integer.")

        # fmt: off
        return {
            "optics_group_name": self.optics_group_name[idx],
            "optics_group_number": self.optics_group_number[idx],
            "micrograph_original_pixel_size": self.micrograph_original_pixel_size[idx],  # noqa: E501
            "micrograph_original_shape_x": self.micrograph_original_shape_x[idx],  # noqa: E501
            "micrograph_original_shape_y": self.micrograph_original_shape_y[idx],  # noqa: E501
            "micrograph_psd_reference": self.micrograph_psd_reference[idx],
            "voltage": self.voltage[idx],
            "spherical_aberration": self.spherical_aberration[idx],
            "amplitude_contrast_ratio": self.amplitude_contrast_ratio[idx],
            "additional_phase_shift": self.additional_phase_shift[idx],
            "image_pixel_size": self.image_pixel_size[idx],
            "image_shape_x": self.image_shape_x[idx],
            "image_shape_y": self.image_shape_y[idx],
            "image_shape_z": self.image_shape_z[idx],
            "image_dimensionality": self.image_dimensionality[idx],
        }
        # fmt: on

    def __add__(self, other: "OpticsGroups") -> "OpticsGroups":
        """Add two OpticsGroups objects together. Effectively concatenates the
        held arrays with checking to ensure names and numbers are unique.
        """
        # Check for duplicate names and numbers
        _dup_names = set(self.optics_group_name) & set(other.optics_group_name)
        _dup_numbers = set(self.optics_group_number) & set(
            other.optics_group_number
        )
        if _dup_names or _dup_numbers:
            raise ValueError(
                "Duplicate optics group names or numbers found in addition. "
                "Please ensure all names and numbers are unique."
            )

        # Concatenate the arrays
        self.optics_group_name.extend(other.optics_group_name)
        self.optics_group_number.extend(other.optics_group_number)
        self.micrograph_original_pixel_size.extend(
            other.micrograph_original_pixel_size
        )
        self.micrograph_original_shape_x.extend(
            other.micrograph_original_shape_x
        )
        self.micrograph_original_shape_y.extend(
            other.micrograph_original_shape_y
        )
        self.micrograph_psd_reference.extend(other.micrograph_psd_reference)
        self.voltage.extend(other.voltage)
        self.spherical_aberration.extend(other.spherical_aberration)
        self.amplitude_contrast_ratio.extend(other.amplitude_contrast_ratio)
        self.additional_phase_shift.extend(other.additional_phase_shift)
        self.defocus_1.extend(other.defocus_1)
        self.defocus_2.extend(other.defocus_2)
        self.astigmatism_azimuth.extend(other.astigmatism_azimuth)
        self.image_pixel_size.extend(other.image_pixel_size)
        self.image_shape_x.extend(other.image_shape_x)
        self.image_shape_y.extend(other.image_shape_y)
        self.image_shape_z.extend(other.image_shape_z)
        self.image_dimensionality.extend(other.image_dimensionality)

        return self

    # TODO: Functionality for adding in a new optics group
    def add_new_from_df(self, df: pd.DataFrame) -> "OpticsGroups":
        """Add a new optics groups (rows) to the OpticsGroups object from
        parameters defined in a DataFrame.

        """
        new_optics_groups = OpticsGroups.from_df(df)
        return self.__add__(new_optics_groups)

    def add_new_from_micrograph(
        self,
        micrograph: "Micrograph",
        box_size: tuple[int, int],
        num_particles: int,
        group_name: str = None,
        group_number: int = 0,
    ) -> "OpticsGroups":
        """Add a new optics group to the OpticsGroups object from a Micrograph
        object. Attempts to update the group number if it already exists.
        """
        # Super simple way of incrementing the group number
        if np.any(self.optics_group_number == group_number):
            group_number = np.max(self.optics_group_number) + 1

        new_optics_groups = OpticsGroups.from_micrograph(
            micrograph, box_size, num_particles, group_name, group_number
        )

        return self.__add__(new_optics_groups)


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

    TODO: Update attributes list

    Methods:
    --------

    """

    # Held per-particle parameters
    particle_coordinates_pixel: np.ndarray
    particle_coordinates_angstrom: np.ndarray
    particle_defocus: np.ndarray
    particle_class_names: np.ndarray
    particle_class_numbers: np.ndarray
    particle_orientations: np.ndarray
    particle_index_in_image_stack: np.ndarray
    particle_image_paths: np.ndarray
    particle_micrograph_paths: np.ndarray
    particle_optics_group_names: np.ndarray
    particle_optics_group_numbers: np.ndarray

    # References to other classes
    optics_groups: Optional[OpticsGroups] = None

    @classmethod
    def from_out_coordinates_and_micrograph(
        cls,
        micrograph: "Micrograph",
        out_coordinates_path: str,
        box_size: tuple[int, int],
        particle_class_numbers: np.ndarray = None,
        particle_class_names: np.ndarray = None,
        image_stack_path: str = None,
        optics_group_name: str = None,
        optics_group_number: int = 0,
        extract_position_reference: Literal["center", "top_left"] = "center",
    ):
        """Create a RefineTemplateResult object from an out_coordinates.txt
        file (produced by cisTEM make_template_result program after
        refine_template) and a Micrograph object as a reference to the original
        image.

        Args:
        -----

        TODO: complete docstring

        Returns:
        --------

        TODO: Figure out way to handle pre-defined PSD classes rather than
        just paths to the PSD files.
        """
        # Validations of the Micrograph object
        assert micrograph.image_array is not None
        assert micrograph.image_path is not None
        assert micrograph.contrast_transfer_function is not None

        # Parse the out_coordinates.txt file into particle parameters dict
        particle_params = _parse_out_coordinates_result_to_particle_params(
            out_coordinates_path, micrograph.pixel_size
        )

        # TODO: better way of handling number of particles
        num_particles = particle_params["particle_coordinates_pixel"].shape[0]

        # Create the image stack from the micrograph and particle parameters
        image_stack, image_stack_path = _extract_particles_from_micrograph(
            particle_params=particle_params,
            micrograph=micrograph,
            box_size=box_size,
            num_particles=num_particles,
            positions_reference=extract_position_reference,
            image_stack_path=image_stack_path,
        )

        particle_params["particle_index_in_image_stack"] = np.arange(
            num_particles
        )
        particle_params["particle_micrograph_paths"] = np.array(
            [micrograph.image_path] * num_particles
        )
        particle_params["particle_image_paths"] = np.array(
            [image_stack_path] * num_particles
        )

        # Create an OpticsGroups object from the micrograph and held CTF
        optics_groups = OpticsGroups.from_micrograph(
            micrograph=micrograph,
            box_size=box_size,
            num_particles=num_particles,
            group_name=optics_group_name,
            group_number=optics_group_number,
        )
        _gp_name = [optics_groups.optics_group_name[0]] * num_particles
        _gp_number = [optics_groups.optics_group_number[0]] * num_particles

        particle_params["optics_groups"] = optics_groups
        particle_params["particle_optics_group_names"] = np.array(_gp_name)
        particle_params["particle_optics_group_numbers"] = np.array(_gp_number)

        # Add the optional particle class names and numbers
        if particle_class_names is not None:
            particle_params["particle_class_names"] = particle_class_names
        else:
            particle_params["particle_class_names"] = np.array(
                [""] * num_particles
            )
        if particle_class_numbers is not None:
            particle_params["particle_class_numbers"] = particle_class_numbers
        else:
            particle_params["particle_class_numbers"] = np.array(
                [-1] * num_particles
            )

        # Finally, instantiate the ParticleStack object
        return cls(**particle_params, image_stack=image_stack)

    def add_new_from_micrograph(
        self,
        micrograph: "Micrograph",
        out_coordinates_path: str,
        box_size: tuple[int, int],
        particle_class_numbers: np.ndarray = None,
        particle_class_names: np.ndarray = None,
        image_stack_path: str = None,
        optics_group_name: str = None,
        optics_group_number: int = 0,
        extract_position_reference: Literal["center", "top_left"] = "center",
    ) -> "ParticleStack":
        """Add a new set of particles to the ParticleStack object from an
        out_coordinates.txt file (produced by cisTEM make_template_result
        program after refine_template) and a Micrograph object as a reference
        to the original image.

        Args:
        -----

        TODO: docstring
        """
        new_particle_stack = ParticleStack.from_out_coordinates_and_micrograph(
            micrograph,
            out_coordinates_path,
            box_size,
            particle_class_numbers,
            particle_class_names,
            image_stack_path,
            optics_group_name,
            optics_group_number,
            extract_position_reference,
        )

        return self + new_particle_stack

    @classmethod
    def from_star(cls, star_path: str):
        """Create a ParticleStack object from a STAR file.

        TODO: complete docstring

        """
        star_dict = starfile.read(star_path, always_dict=True)
        optics_df = star_dict["optics"]
        part_df = star_dict["particles"]

        # Parse the optics dataframe into an OpticsGroups object
        optics_groups = OpticsGroups.from_df(optics_df)

        # Extract the necessary information from the particles dataframe
        particle_coordinates_pixel = np.stack(
            (
                part_df["mosaicsCoordinateX"].values,
                part_df["mosaicsCoordinateY"].values,
            ),
            axis=-1,
        )
        particle_coordinates_angstrom = np.stack(
            (
                part_df["mosaicsCoordinateXAngstrom"].values,
                part_df["mosaicsCoordinateYAngstrom"].values,
            ),
            axis=-1,
        )
        particle_defocus = part_df["mosaicsParticleDefocusAngstrom"].values
        particle_class_names = part_df["mosaicsParticleClassName"].values
        particle_class_numbers = part_df["mosaicsParticleClassNumber"].values
        particle_orientations = np.stack(
            (
                part_df["mosaicsOrientationPhi"].values,
                part_df["mosaicsOrientationTheta"].values,
                part_df["mosaicsOrientationPsi"].values,
            ),
            axis=-1,
        )
        particle_index_in_image_stack = part_df[
            "mosaicsIndexInImageStack"
        ].values
        particle_image_paths = part_df["mosaicsParticleImagePath"].values
        particle_micrograph_paths = part_df["mosaicsMicrographPath"].values
        particle_optics_group_names = part_df["mosaicsOpticsGroupName"].values
        particle_optics_group_numbers = part_df[
            "mosaicsOpticsGroupNumber"
        ].values

        # TODO validating of the parsed STAR values

        return cls(
            optics_groups=optics_groups,
            particle_coordinates_pixel=particle_coordinates_pixel,
            particle_coordinates_angstrom=particle_coordinates_angstrom,
            particle_defocus=particle_defocus,
            particle_class_names=particle_class_names,
            particle_class_numbers=particle_class_numbers,
            particle_orientations=particle_orientations,
            particle_index_in_image_stack=particle_index_in_image_stack,
            particle_image_paths=particle_image_paths,
            particle_micrograph_paths=particle_micrograph_paths,
            particle_optics_group_names=particle_optics_group_names,
            particle_optics_group_numbers=particle_optics_group_numbers,
        )

    def to_star(self, star_path: str, mrcs_path: str):
        """Write the ParticleStack object to a STAR file.

        TODO: complete docstring
        """
        if self.optics_groups is None:
            raise ValueError("OpticsGroups object not found in ParticleStack.")
        optics_df = self.optics_groups.to_df()

        # Create the particles dataframe
        part_df = pd.DataFrame()
        part_df["mosaicsCoordinateX"] = self.particle_coordinates_pixel[:, 0]
        part_df["mosaicsCoordinateY"] = self.particle_coordinates_pixel[:, 1]
        part_df["mosaicsCoordinateXAngstrom"] = (
            self.particle_coordinates_angstrom[:, 0]
        )
        part_df["mosaicsCoordinateYAngstrom"] = (
            self.particle_coordinates_angstrom[:, 1]
        )
        part_df["mosaicsParticleDefocusAngstrom"] = self.particle_defocus
        part_df["mosaicsParticleClassName"] = self.particle_class_names
        part_df["mosaicsParticleClassNumber"] = self.particle_class_numbers
        part_df["mosaicsOrientationPhi"] = self.particle_orientations[:, 0]
        part_df["mosaicsOrientationTheta"] = self.particle_orientations[:, 1]
        part_df["mosaicsOrientationPsi"] = self.particle_orientations[:, 2]
        part_df["mosaicsIndexInImageStack"] = (
            self.particle_index_in_image_stack
        )
        part_df["mosaicsParticleImagePath"] = self.particle_image_paths
        part_df["mosaicsMicrographPath"] = self.particle_micrograph_paths
        part_df["mosaicsOpticsGroupName"] = self.particle_optics_group_names
        part_df["mosaicsOpticsGroupNumber"] = (
            self.particle_optics_group_numbers
        )

        # TODO: write .mrcs files for image stack
        _ = mrcs_path

        starfile.write({"optics": optics_df, "particles": part_df}, star_path)

    def __init__(
        self,
        particle_coordinates_pixel: np.ndarray,
        particle_coordinates_angstrom: np.ndarray,
        particle_defocus: np.ndarray,
        particle_class_names: np.ndarray,
        particle_class_numbers: np.ndarray,
        particle_orientations: np.ndarray,
        particle_index_in_image_stack: np.ndarray,
        particle_image_paths: np.ndarray,
        particle_micrograph_paths: np.ndarray,
        particle_optics_group_names: np.ndarray,
        particle_optics_group_numbers: np.ndarray,
        optics_groups: Optional[OpticsGroups] = None,
        image_stack: Optional[np.ndarray] = None,
    ):
        self.particle_coordinates_pixel = particle_coordinates_pixel
        self.particle_coordinates_angstrom = particle_coordinates_angstrom
        self.particle_defocus = particle_defocus
        self.particle_class_names = particle_class_names
        self.particle_class_numbers = particle_class_numbers
        self.particle_orientations = particle_orientations
        self.particle_index_in_image_stack = particle_index_in_image_stack
        self.particle_image_paths = particle_image_paths
        self.particle_micrograph_paths = particle_micrograph_paths
        self.particle_optics_group_names = particle_optics_group_names
        self.particle_optics_group_numbers = particle_optics_group_numbers

        self.optics_groups = optics_groups

        # TODO: Validation for passed values (e.g. auto-extract particles)
        if image_stack is None and np.any(
            (particle_image_paths is None) | (particle_image_paths == "")
        ):
            raise NotImplementedError("Auto particle extract not implemented.")

        self.image_stack = image_stack

    def __add__(self, other: "ParticleStack") -> "ParticleStack":
        """Adds two ParticleStack objects together. This is done by
        concatenating the held arrays and updating the particle index in the
        image stack. Also adds the held OpticsGroups objects together.
        """

        @typing.no_type_check
        def _handle_optics_groups_addition(self, other):
            # TODO: Better way of handling this logic (prob helper function)
            # type: ignore
            _self_has_optics = self.optics_groups is not None
            _other_has_optics = other.optics_groups is not None
            if _self_has_optics and _other_has_optics:
                # TODO: Move group number checking to OpticsGroups class later
                if np.any(
                    set(self.optics_groups.optics_group_number)
                    & set(other.optics_groups.optics_group_number)
                ):
                    _inc = np.max(self.optics_groups.optics_group_number) + 1
                    other.optics_groups.optics_group_number = [
                        x + _inc
                        for x in other.optics_groups.optics_group_number
                    ]
                    other.particle_optics_group_numbers += _inc

                # Add the OpticsGroups objects together
                self.optics_groups = self.optics_groups + other.optics_groups
            elif _self_has_optics and not _other_has_optics:
                raise ValueError("OpticsGroups object not found in other.")
            elif not _self_has_optics and _other_has_optics:
                raise ValueError("OpticsGroups object not found in self.")
            # type: ignore

        # Concatenate the ParticleStack arrays together
        self.particle_coordinates_pixel = np.concatenate(
            (self.particle_coordinates_pixel, other.particle_coordinates_pixel)
        )
        self.particle_coordinates_angstrom = np.concatenate(
            (
                self.particle_coordinates_angstrom,
                other.particle_coordinates_angstrom,
            )
        )
        self.particle_defocus = np.concatenate(
            (self.particle_defocus, other.particle_defocus)
        )
        self.particle_class_names = np.concatenate(
            (self.particle_class_names, other.particle_class_names)
        )
        self.particle_class_numbers = np.concatenate(
            (self.particle_class_numbers, other.particle_class_numbers)
        )
        self.particle_orientations = np.concatenate(
            (self.particle_orientations, other.particle_orientations)
        )
        self.particle_index_in_image_stack = np.concatenate(
            (
                self.particle_index_in_image_stack,
                other.particle_index_in_image_stack,
            )
        )
        self.particle_image_paths = np.concatenate(
            (self.particle_image_paths, other.particle_image_paths)
        )
        self.particle_micrograph_paths = np.concatenate(
            (self.particle_micrograph_paths, other.particle_micrograph_paths)
        )
        self.particle_optics_group_names = np.concatenate(
            (
                self.particle_optics_group_names,
                other.particle_optics_group_names,
            )
        )
        self.particle_optics_group_numbers = np.concatenate(
            (
                self.particle_optics_group_numbers,
                other.particle_optics_group_numbers,
            )
        )

        # Concatenate the image stack
        self.image_stack = np.concatenate(
            (self.image_stack, other.image_stack)
        )

        return self


def _parse_out_coordinates_result_to_particle_params(
    out_coordinates_path: str, pixel_size: float
) -> dict[str, np.ndarray]:
    """Given a path to an out_coordinates.txt file generated by cisTEM,
    parse the file into the coordinates, defocus, and orientations of the
    particles. Returns a dictionary where the keys correspond to attribute
    names of the ParticleStack class and values are the parsed arrays.

    Args:
    -----

    out_coordinates_path (str): Path to the out_coordinates.txt file.
    pixel_size (float): Pixel size of the micrograph in Angstroms.

    Returns:
    --------

    dict[str, np.ndarray]: Dictionary of parsed particle parameters.
    """
    coord_df = parse_out_coordinates_result(out_coordinates_path)

    # Get pixel coordinates for each of the particles
    coordinates_x_angstrom = coord_df["X"].values
    coordinates_y_angstrom = coord_df["Y"].values
    coordinates_angstrom = np.stack(
        (coordinates_x_angstrom, coordinates_y_angstrom), axis=-1
    )
    coordinates_pixel = np.round(coordinates_angstrom / pixel_size).astype(int)

    defocus = coord_df["Z"].values

    # Get orientations for each of the particles
    orientations = np.stack(
        (
            coord_df["Psi"].values,
            coord_df["Theta"].values,
            coord_df["Phi"].values,
        ),
        axis=-1,
    )

    return {
        "particle_coordinates_pixel": coordinates_pixel,
        "particle_coordinates_angstrom": coordinates_angstrom,
        "particle_defocus": defocus,
        "particle_orientations": orientations,
    }


def _extract_particles_from_micrograph(
    particle_params: dict[str, np.ndarray],
    micrograph: "Micrograph",
    box_size: tuple[int, int],
    num_particles,
    positions_reference: Literal["center", "top_left"],
    image_stack_path: Optional[str] = None,
) -> tuple[np.ndarray, str]:
    """Given a set of particle parameters and a Micrograph object, extract the
    particle images from the

    TODO: complete docstring
    """
    # Determine the image stack path, if not provided
    if image_stack_path is None and micrograph.image_path is not None:
        basename = str(os.path.basename(micrograph.image_path))
        dirname = str(os.path.dirname(micrograph.image_path))

        image_stack_path = f"{dirname}/{basename[:-4]}_stack.mrcs"
    elif image_stack_path is None:
        raise ValueError("Cannot auto-resolve path for image_stack.")

    # Get the pixel coordinates for the particles
    coordinates_pixel = particle_params["particle_coordinates_pixel"]

    # Extract the particles from the micrograph
    image_stack = np.zeros((num_particles, *box_size))
    for i, (x, y) in enumerate(coordinates_pixel):
        image_stack[i] = get_cropped_region_of_image(
            image=micrograph.image_array,
            box_size=box_size,
            pos_x=x,
            pos_y=y,
            positions_reference=positions_reference,
        )

    return image_stack, image_stack_path
