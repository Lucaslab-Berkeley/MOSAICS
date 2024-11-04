from mosaics.data_structures.contrast_transfer_function import \
    ContrastTransferFunction
from mosaics.data_structures.micrograph import Micrograph
from mosaics.data_structures.particle_stack import ParticleStack


class MosaicsManager:
    """Manager class for handling the pipeline of operations for running
    MOSAICS. The manager holds a reference to a Micrograph object and/or a
    ParticleStack object and...

    TODO: Complete docstring

    """

    micrograph: Micrograph
    particle_stack: ParticleStack
    ctf: ContrastTransferFunction
