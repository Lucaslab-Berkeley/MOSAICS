"""Manager class for running MOSAICS."""

from leopard_em.pydantic_models import ParticleStack
from pydantic import BaseModel
from ttsim3d.models import Simulator

from .mosaics_result import MosaicsResult
from .template_iterator import BaseTemplateIterator


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
    template_iterator: BaseTemplateIterator
    mosaics_result: MosaicsResult

    def run_mosaics(self) -> None:
        """Run the MOSAICS program.

        TODO: Complete docstring
        """
        raise NotImplementedError("MOSAICS program not yet implemented.")
