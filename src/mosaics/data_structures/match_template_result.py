import numpy as np

from mosaics.data_structures.reference_template import ReferenceTemplate
from mosaics.data_structures.micrograph import Micrograph


class MatchTemplateResult:
    """Class which holds references to a micrograph and the results of a template
    matching job.
    
    TODO: Incorporate functionality to load entire job outputs only temporarily deleting
    after use. Saves memory and avoids I/O bottleneck. Use the mrcfile package
    
    TODO: complete docstring
    """
    
    reference_template: ReferenceTemplate
    micrograph: Micrograph
    
    maximum_intensity_projection: np.ndarray
    cross_correlation_average: np.ndarray
    cross_correlation_variance: np.ndarray
    scaled_mip: np.ndarray
    
    best_phi: np.ndarray
    best_theta: np.ndarray
    best_psi: np.ndarray
    best_defocus: np.ndarray  # Relative to micrograph defocus
    
    # Paths to output files
    maximum_intensity_projection_path: str
    cross_correlation_average_path: str
    cross_correlation_variance_path: str
    scaled_mip_path: str
    best_phi_path: str
    best_theta_path: str
    best_psi_path: str
    best_defocus_path: str
    
    @classmethod
    def from_match_template_output_files(
        cls,
        reference_template: ReferenceTemplate,
        micrograph: Micrograph,
        maximum_intensity_projection_path: str,
        cross_correlation_average_path: str,
        cross_correlation_variance_path: str,
        scaled_mip_path: str,
        best_phi_path: str,
        best_theta_path: str,
        best_psi_path: str,
        best_defocus_path: str,
    ):
        """Create a MatchTemplateResult from the output files of a template matching
        job.
        """
        # TODO: handle both mrc and numpy files intelligently
        
        return cls(
            reference_template=reference_template,
            micrograph=micrograph,
            maximum_intensity_projection=np.load(maximum_intensity_projection_path),
            cross_correlation_average=np.load(cross_correlation_average_path),
            cross_correlation_variance=np.load(cross_correlation_variance_path),
            scaled_mip=np.load(scaled_mip_path),
            best_phi=np.load(best_phi_path),
            best_theta=np.load(best_theta_path),
            best_psi=np.load(best_psi_path),
            best_defocus=np.load(best_defocus_path),
        )
    
    def __init__(
        self,
        reference_template: ReferenceTemplate,
        micrograph: Micrograph,
        maximum_intensity_projection: np.ndarray,
        cross_correlation_average: np.ndarray,
        cross_correlation_variance: np.ndarray,
        scaled_mip: np.ndarray,
        best_phi: np.ndarray,
        best_theta: np.ndarray,
        best_psi: np.ndarray,
        best_defocus: np.ndarray,
        maximum_intensity_projection_path: str = None,
        cross_correlation_average_path: str = None,
        cross_correlation_variance_path: str = None,
        scaled_mip_path: str = None,
        best_phi_path: str = None,
        best_theta_path: str = None,
        best_psi_path: str = None,
        best_defocus_path: str = None,
    ):
        self.reference_template = reference_template
        self.micrograph = micrograph
        self.maximum_intensity_projection = maximum_intensity_projection
        self.cross_correlation_average = cross_correlation_average
        self.cross_correlation_variance = cross_correlation_variance
        self.scaled_mip = scaled_mip
        self.best_phi = best_phi
        self.best_theta = best_theta
        self.best_psi = best_psi
        self.best_defocus = best_defocus
        
        self.maximum_intensity_projection_path = maximum_intensity_projection_path
        self.cross_correlation_average_path = cross_correlation_average_path
        self.cross_correlation_variance_path = cross_correlation_variance_path
        self.scaled_mip_path = scaled_mip_path
        self.best_phi_path = best_phi_path
        self.best_theta_path = best_theta_path
        self.best_psi_path = best_psi_path
        self.best_defocus_path = best_defocus_path

    def to_json(self) -> dict:
        """Convert the MatchTemplateResult object to a JSON-serializable dictionary."""
        return {
            "reference_template": self.reference_template.to_json(),
            "micrograph": self.micrograph.to_json(),
            "maximum_intensity_projection_path": self.maximum_intensity_projection_path,
            "cross_correlation_average_path": self.cross_correlation_average_path,
            "cross_correlation_variance_path": self.cross_correlation_variance_path,
            "scaled_mip_path": self.scaled_mip_path,
            "best_phi_path": self.best_phi_path,
            "best_theta_path": self.best_theta_path,
            "best_psi_path": self.best_psi_path,
            "best_defocus_path": self.best_defocus_path,
        }

    @classmethod
    def from_json(cls, json_dict: dict) -> "MatchTemplateResult":
        """Create a MatchTemplateResult object from a JSON dictionary."""
        reference_template = ReferenceTemplate.from_json(json_dict["reference_template"])
        micrograph = Micrograph.from_json(json_dict["micrograph"])
        
        return cls(
            reference_template=reference_template,
            micrograph=micrograph,
            maximum_intensity_projection=np.load(json_dict["maximum_intensity_projection_path"]),
            cross_correlation_average=np.load(json_dict["cross_correlation_average_path"]),
            cross_correlation_variance=np.load(json_dict["cross_correlation_variance_path"]),
            scaled_mip=np.load(json_dict["scaled_mip_path"]),
            best_phi=np.load(json_dict["best_phi_path"]),
            best_theta=np.load(json_dict["best_theta_path"]),
            best_psi=np.load(json_dict["best_psi_path"]),
            best_defocus=np.load(json_dict["best_defocus_path"]),
            maximum_intensity_projection_path=json_dict["maximum_intensity_projection_path"],
            cross_correlation_average_path=json_dict["cross_correlation_average_path"],
            cross_correlation_variance_path=json_dict["cross_correlation_variance_path"],
            scaled_mip_path=json_dict["scaled_mip_path"],
            best_phi_path=json_dict["best_phi_path"],
            best_theta_path=json_dict["best_theta_path"],
            best_psi_path=json_dict["best_psi_path"],
            best_defocus_path=json_dict["best_defocus_path"],
        )
