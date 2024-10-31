import numpy as np
import mrcfile
import json


class RefineTemplateResult:
    """Class to store references to the refine template result files and generate
    ParticleStack objects.
    
    TODO: complete docstring
    """
    
    reference_template: "ReferenceTemplate"
    micrograph: "Micrograph"
    
    num_particles: int
    z_score_threshold: float  # TODO: Auto-determine from tm results
    particle_positions: np.ndarray  # In pixel coordinates
    
    # NOTE: These are 1D arrays where the output for cisTEM refine_template program is
    # 2D arrays of same shape of image.
    refined_mip: np.ndarray
    refined_mip_scaled: np.ndarray
    refined_phi: np.ndarray
    refined_theta: np.ndarray
    refined_psi: np.ndarray
    refined_defocus: np.ndarray  # Relative to micrograph defocus, isotropic across axes
    
    # Paths to output files
    refined_mip_path: str
    refined_mip_scaled_path: str
    refined_phi_path: str
    refined_theta_path: str
    refined_psi_path: str
    refined_defocus_path: str
    
    @classmethod
    def from_refine_template_output_files(
        cls,
        reference_template: "ReferenceTemplate",
        micrograph: "Micrograph",
        z_score_threshold: float,
        refined_mip_path: str,
        refined_mip_scaled_path: str,
        refined_psi_path: str,
        refined_theta_path: str,
        refined_phi_path: str,
        refined_defocus_path: str,
    ):
        """Create a RefineTemplateResult object from the output files of the cisTEM
        refine_template program.
        
        TODO: complete docstring
        """
        # Open the MRC files and ignore last dimension if 3d
        refined_mip_2d = mrcfile.open(refined_mip_path).data.copy().squeeze()
        refined_phi = mrcfile.open(refined_phi_path).data.copy().squeeze()
        refined_theta = mrcfile.open(refined_theta_path).data.copy().squeeze()
        refined_psi = mrcfile.open(refined_psi_path).data.copy().squeeze()
        refined_defocus = mrcfile.open(refined_defocus_path).data.copy().squeeze()
        refined_mip_scaled = mrcfile.open(refined_mip_scaled_path).data.copy().squeeze()
        
        # Find where the mip is non-zero for locating each particle.
        particle_positions = np.where(refined_mip_2d != 0)
        num_particles = particle_positions[0].size
        
        # Index the 2D arrays into 1D arrays at the particle locations
        refined_mip = refined_mip_2d[particle_positions]
        refined_phi = refined_phi[particle_positions]
        refined_theta = refined_theta[particle_positions]
        refined_psi = refined_psi[particle_positions]
        refined_defocus = refined_defocus[particle_positions]
        refined_mip_scaled = refined_mip_scaled[particle_positions]
        
        # Cast particle locations into array
        particle_positions = np.stack([particle_positions[0], particle_positions[1]], axis=-1)
        
        return cls(
            reference_template=reference_template,
            micrograph=micrograph,
            z_score_threshold=z_score_threshold,
            num_particles=num_particles,
            particle_positions=particle_positions,  # TODO: check array shape/orientation
            refined_mip=refined_mip,
            refined_mip_scaled=refined_mip_scaled,
            refined_phi=refined_phi,
            refined_theta=refined_theta,
            refined_psi=refined_psi,
            refined_defocus=refined_defocus,
            refined_mip_path=refined_mip_path,
            refined_mip_scaled_path=refined_mip_scaled_path,
            refined_phi_path=refined_phi_path,
            refined_theta_path=refined_theta_path,
            refined_psi_path=refined_psi_path,
            refined_defocus_path=refined_defocus_path,
        )
        
    def __init__(
        self,
        reference_template: "ReferenceTemplate",
        micrograph: "Micrograph",
        z_score_threshold: float,
        num_particles: int,
        particle_positions: np.ndarray,
        refined_mip: np.ndarray,
        refined_mip_scaled: np.ndarray,
        refined_phi: np.ndarray,
        refined_theta: np.ndarray,
        refined_psi: np.ndarray,
        refined_defocus: np.ndarray,
        refined_mip_path: str = None,
        refined_mip_scaled_path: str = None,
        refined_phi_path: str = None,
        refined_theta_path: str = None,
        refined_psi_path: str = None,
        refined_defocus_path: str = None,
    ):
        self.reference_template = reference_template
        self.micrograph = micrograph
        
        self.z_score_threshold = z_score_threshold
        self.num_particles = num_particles
        self.particle_positions = particle_positions
        
        self.refined_mip = refined_mip
        self.refined_mip_scaled = refined_mip_scaled
        self.refined_phi = refined_phi
        self.refined_theta = refined_theta
        self.refined_psi = refined_psi
        self.refined_defocus = refined_defocus
        
        self.refined_mip_path = refined_mip_path
        self.refined_mip_scaled_path = refined_mip_scaled_path
        self.refined_phi_path = refined_phi_path
        self.refined_theta_path = refined_theta_path
        self.refined_psi_path = refined_psi_path
        self.refined_defocus_path = refined_defocus_path
        
    def to_json(self) -> dict:
        """Convert the RefineTemplateResult object to a JSON-serializable dictionary.
        Large arrays are not included, only their file paths are stored."""
        return {
            "reference_template": self.reference_template.to_json(),
            "micrograph": self.micrograph.to_json(),
            "z_score_threshold": self.z_score_threshold,
            "num_particles": self.num_particles,
            "refined_mip_path": self.refined_mip_path,
            "refined_mip_scaled_path": self.refined_mip_scaled_path,
            "refined_phi_path": self.refined_phi_path,
            "refined_theta_path": self.refined_theta_path,
            "refined_psi_path": self.refined_psi_path,
            "refined_defocus_path": self.refined_defocus_path,
        }

    @classmethod
    def from_json(cls, json_dict: dict) -> "RefineTemplateResult":
        """Create a RefineTemplateResult object from a JSON dictionary.
        Arrays are loaded from the stored file paths."""
        # Load reference objects
        reference_template = ReferenceTemplate.from_json(json_dict["reference_template"])
        micrograph = Micrograph.from_json(json_dict["micrograph"])
        
        # Load arrays from paths
        refined_mip_2d = np.load(json_dict["refined_mip_path"])
        particle_positions = np.where(refined_mip_2d != 0)
        num_particles = particle_positions[0].size
        
        refined_mip = refined_mip_2d[particle_positions]
        refined_mip_scaled = np.load(json_dict["refined_mip_scaled_path"])[particle_positions]
        refined_phi = np.load(json_dict["refined_phi_path"])[particle_positions]
        refined_theta = np.load(json_dict["refined_theta_path"])[particle_positions]
        refined_psi = np.load(json_dict["refined_psi_path"])[particle_positions]
        refined_defocus = np.load(json_dict["refined_defocus_path"])[particle_positions]

        return cls(
            reference_template=reference_template,
            micrograph=micrograph,
            z_score_threshold=json_dict["z_score_threshold"],
            num_particles=num_particles,
            particle_positions=particle_positions,
            refined_mip=refined_mip,
            refined_mip_scaled=refined_mip_scaled,
            refined_phi=refined_phi,
            refined_theta=refined_theta,
            refined_psi=refined_psi,
            refined_defocus=refined_defocus,
            refined_mip_path=json_dict["refined_mip_path"],
            refined_mip_scaled_path=json_dict["refined_mip_scaled_path"],
            refined_phi_path=json_dict["refined_phi_path"],
            refined_theta_path=json_dict["refined_theta_path"],
            refined_psi_path=json_dict["refined_psi_path"],
            refined_defocus_path=json_dict["refined_defocus_path"],
        )
        
