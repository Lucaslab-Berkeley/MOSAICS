"""Manager class for running MOSAICS."""

import mmdf
import yaml  # type: ignore
from leopard_em.pydantic_models import ParticleStack
from pydantic import BaseModel
from ttsim3d.models import Simulator

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
    mosaics_results : MosaicsResults
        Results model for storing the output of the MOSAICS program.
    """

    particle_stack: ParticleStack  # comes from Leopard-EM
    simulator: Simulator  # comes from ttsim3d
    template_iterator: ResidueTemplateIterator
    # mosaics_result: MosaicsResult

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

    def run_mosaics(self) -> None:
        """Run the MOSAICS program.

        TODO: Complete docstring
        """
        raise NotImplementedError("MOSAICS program not yet implemented.")
