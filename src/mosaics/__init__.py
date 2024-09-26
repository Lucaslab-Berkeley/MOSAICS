import numpy as np
from .micrograph import Micrograph
from .particle_stack import ParticleStack
from .filters.contrast_transfer_function import ContrastTransferFunction


class MosaicsManager:
    """Manager class for handling the pipeline of operations for running MOSAICS. The
    manager holds a reference to a Micrograph object and/or a ParticleStack object
    TODO: Complete docstring
    
    """
    
    micrograph: Micrograph
    particle_stack: ParticleStack
    ctf: ContrastTransferFunction
    
    