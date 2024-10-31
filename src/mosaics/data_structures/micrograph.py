import numpy as np
import mrcfile
from typing import Tuple
from typing import Literal

from mosaics.data_structures.particle_stack import ParticleStack
from mosaics.data_structures.contrast_transfer_function import ContrastTransferFunction
from mosaics.utils import get_cropped_region_of_image

class Micrograph:
    """Class for handling micrograph data and common operations.

    Attributes: TODO

    Methods: TODO

    """

    image_array: np.ndarray
    pixel_size: float  # In Angstroms, assume square pixels
    image_path: str

    ctf: "ContrastTransferFunction" = None

    @classmethod
    def from_mrc(cls, mrc_path: str):
        """Create a Micrograph object from an MRC file."""
        with mrcfile.open(mrc_path) as mrc:
            image_array = mrc.data.squeeze().copy()
            pixel_size = mrc.voxel_size.x
            
        return cls(image_array, pixel_size, mrc_path)
        
    def to_json(self) -> dict:
        """Convert the Micrograph object to a JSON-serializable dictionary."""
        return {
            "pixel_size": self.pixel_size,
            "image_path": self.image_path,
            "ctf": self.ctf.to_json() if self.ctf is not None else None,
        }

    @classmethod
    def from_json(cls, json_dict: dict) -> "Micrograph":
        """Create a Micrograph object from a JSON dictionary."""
        # Load the image from the stored path
        with mrcfile.open(json_dict["image_path"]) as mrc:
            image_array = mrc.data.copy()
        
        micrograph = cls(
            image_array=image_array,
            pixel_size=json_dict["pixel_size"],
            image_path=json_dict["image_path"]
        )
        
        if json_dict["ctf"] is not None:
            micrograph.ctf = ContrastTransferFunction.from_json(json_dict["ctf"])
        
        return micrograph

    def __init__(
        self, image_array: np.ndarray, pixel_size: float, image_path: str = None
    ):
        self.image_array = image_array
        self.pixel_size = pixel_size
        self.image_path = image_path

    def _validate_to_particle_stack_inputs(
        self,
        positions_x: np.ndarray,
        positions_y: np.ndarray,
        particle_orientations: np.ndarray = None,
        particle_defocus_parameters: np.ndarray = None,
        particle_z_scores: np.ndarray = None,
        particle_mip_values: np.ndarray = None,
    ) -> None:
        """Helper function to validate inputs for particle extraction.

        Checks that the number of particles is consistent across all provided arrays and
        that the shapes of the arrays match what is expected.

        Args:
            particle_positions: Array of particle positions, shape (N,2)
            particle_orientations: Optional array of orientations, shape (N,3)
            particle_defocus_parameters: Optional array of CTF params, shape (N,3)
            particle_z_scores: Optional array of z-scores, shape (N,)
            particle_mip_values: Optional array of MIP values, shape (N,)
        """
        assert (
            positions_x.shape == positions_y.shape
        ), "Positions x and y must have the same shape."
        
        # Same length validation
        if particle_orientations is not None:
            assert (
                positions_x.shape[0] == particle_orientations.shape[0]
            ), "Number of particle positions and orientations must be equal."

        if particle_defocus_parameters is not None:
            assert (
                positions_x.shape[0] == particle_defocus_parameters.shape[0]
            ), "Number of particle positions and CTF parameters must be equal."

        if particle_z_scores is not None:
            assert (
                positions_x.shape[0] == particle_z_scores.shape[0]
            ), "Number of particle positions and z-scores must be equal."

        if particle_mip_values is not None:
            assert (
                positions_x.shape[0] == particle_mip_values.shape[0]
            ), "Number of particle positions and MIP values must be equal."

        # Shape validation
        if particle_orientations is not None:
            assert (
                particle_orientations.shape[1] == 3
            ), "Orientation array must have 3 columns."

        if particle_defocus_parameters is not None:
            assert (
                particle_defocus_parameters.shape[1] == 3
            ), "Defocus parameter array must have 3 columns."

        if particle_z_scores is not None:
            assert particle_z_scores.ndim == 1, "Z-score array must be 1-dimensional."

        if particle_mip_values is not None:
            assert (
                particle_mip_values.ndim == 1
            ), "MIP value array must be 1-dimensional."

    def to_particle_stack(
        self,
        box_size: Tuple[int, int],
        positions_x: np.ndarray,
        positions_y: np.ndarray,
        positions_reference: Literal["center", "corner"] = "center",
        handle_bounds: Literal["crop", "fill", "error"] = "error",
        particle_orientations: np.ndarray = None,
        particle_defocus_parameters: np.ndarray = None,
        particle_z_scores: np.ndarray = None,
        particle_mip_values: np.ndarray = None,
    ) -> "ParticleStack":
        """Extract particles from the micrograph using the provided particle positions
        and other optional information about each particle.

        Args:
        -----

        TODO: Args

        Returns:
        --------

            ParticleStack: A ParticleStack object containing the extracted particles.

        """
        self._validate_to_particle_stack_inputs(
            positions_x,
            positions_y,
            particle_orientations,
            particle_defocus_parameters,
            particle_z_scores,
            particle_mip_values,
        )


        # Iterate over each position and extract the particle
        particle_images = []
        for i in range(positions_x.shape[0]):
            particle_images.append(
                get_cropped_region_of_image(
                    self.image_array,
                    box_size,
                    positions_x[i],
                    positions_y[i],
                    positions_reference,
                    handle_bounds,
                )
            )
            
        # DEBUG make mosaic of particle images
        nrows = int(np.ceil(np.sqrt(len(particle_images))))
        ncols = int(np.ceil(len(particle_images) / nrows))

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows, ncols, figsize=(24, 24))
        
        for i, img in enumerate(particle_images):
            row = i // ncols
            col = i % ncols
            axs[row, col].imshow(img, cmap="gray")
            
        plt.show()
            
        particle_images = np.array(particle_images)
            
        # Convert x, y positions to a single numpy array
        particle_positions = np.vstack((positions_x, positions_y)).T

        return ParticleStack(
            pixel_size=self.pixel_size,
            box_size=box_size,
            particle_images=particle_images,
            particle_positions=particle_positions,
            particle_orientations=particle_orientations,
            particle_defocus_parameters=particle_defocus_parameters,
            particle_z_scores=particle_z_scores,
            particle_mip_values=particle_mip_values,
            micrograph_reference_paths=[self.image_path]
        )

    # def plot_particle_boxes(
    #     self,
    #     box_size: Tuple[int, int],
    #     particle_positions: np.ndarray,
    #     particle_orientations: np.ndarray = None,  # Unused but kept for consistency
    #     particle_defocus_parameters: np.ndarray = None,  # Unused but kept for consistency
    #     particle_z_scores: np.ndarray = None,  # Unused but kept for consistency
    #     particle_mip_values: np.ndarray = None,  # Unused but kept for consistency
    #     position_reference: Literal["center", "corner"] = "center",
    # ) -> None:
    #     """Plot the micrograph with red boxes indicating particle positions.
        
    #     Args match to_particle_stack() for convenience. Only box_size, particle_positions,
    #     and position_reference are used.
    #     """
    #     import matplotlib.pyplot as plt
    #     from matplotlib.patches import Rectangle
        
    #     # Adjust positions if reference point is centered
    #     positions = particle_positions.copy()
    #     if position_reference == "center":
    #         positions = positions - np.array(box_size) // 2
        
    #     tmp = self.image_array
    #     pts = np.where(tmp > 7.85)
    #     tmp = gaussian_filter(tmp, sigma=5)
        
    #     # Create the plot
    #     fig, ax = plt.subplots(figsize=(24, 24))
    #     ax.imshow(tmp, cmap="gray")
        
    #     # Scatter where image is above threshold
    #     # TODO: Remove later
    #     ax.scatter(pts[1], pts[0], color="red", s=4, alpha=0.5)
    #     ax.scatter(positions[:, 1], positions[:, 0], color="blue", s=10, marker="x", alpha=0.5)
        
    #     # # Add boxes for each particle
    #     # for pos in positions:
    #     #     x, y = pos
    #     #     rect = Rectangle(
    #     #         (x, y),
    #     #         box_size[1],  # width
    #     #         box_size[0],  # height
    #     #         fill=False,
    #     #         color="red",
    #     #         linewidth=1
    #     #     )
    #     #     ax.add_patch(rect)
        
    #     plt.show()
