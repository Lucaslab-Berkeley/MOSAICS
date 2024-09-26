import numpy as np
import mrcfile
import starfile

from typing import List
from typing import Literal
from typing import Tuple
from typing import Union

# TODO: importing from pre-defined particle stack .mrc file and .star file


class ParticleStack:
    """Class for handing a stack of extracted particle images and their orientations.

    Attributes:
    -----------
    pixel_size: float
        The size of a single pixel, in units of Angstroms.
    particle_images_array: np.ndarray
        A 3D numpy array of shape (N, H, W) where N is the number of particles, H is
        the height of the particle image, and W is the width of the particle image.
    orientations: np.ndarray
        A 2D numpy array of shape (N, 3) where N is the number of particles, and each
        element are the Euler angles (phi, theta, psi) for the particle orientation in
        ZYZ convention.
    particle_positions: np.ndarray
        A 2D numpy array of shape (N, 2) where N is the number of particles, and each
        element are the (x, y) pixel coordinates of the particle in the micrograph.
    particle_defocus_parameters: np.ndarray
        A 2D numpy array of shape (N, 4) where N is the number of particles, and each
        element are the defocus parameters (defocus_1, defocus_2, astigmatism angle,
        phase shift) for the particle.

    Methods:
    --------
    from_mrc_and_star(cls, mrc_file, star_file, star_file_type):
        Class method to instantiate a ParticleStack object from a .mrc file and a .star
        file holding the particle images and their orientations, respectively.

    """

    pixel_size: float
    particle_images_array: np.ndarray
    particle_orientations: np.ndarray
    particle_positions: np.ndarray
    particle_defocus_parameters: np.ndarray

    @classmethod
    def from_mrc_and_star(
        cls, mrc_file: str, star_file: str, star_file_type: Literal["relion", "cisTEM"]
    ):
        """Create a particle stack object from a .mrc file and a .star file. Currently
        supported .star file types are "relion" and "cisTEM".
        """
        # Extract the particle orientations from the .star file
        if star_file_type == "relion":
            raise NotImplementedError
        elif star_file_type == "cisTEM":
            df = starfile.open(star_file)
            
            # TODO: Grab the correct columns name 
            phi = df["cisTEM_phi"].values
            theta = df["cisTEM_theta"].values
            psi = df["cisTEM_psi"].values
            
        else:
            raise ValueError(f"Invalid star file type: {star_file_type}")

    def __init__(self, pixel_size, particle_images_array, particle_orientations, particle_positions, particle_defocus_parameters = None):
        self.particle_images_array = particle_images_array
        self.orientations = orientations
