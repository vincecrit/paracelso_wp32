"""
This module provides a factory for creating instances of various optical flow algorithms.
It defines an enumeration `AlgFactory` that maps algorithm names to their corresponding classes,
and a function `get_algorithm` to retrieve an algorithm instance by its name.
Classes:
    AlgFactory (Enum): Factory class for creating algorithm instances.
Functions:
    get_algorithm(algname: str) -> interfaces.OTAlgorithm: Get an algorithm instance by name.
"""
from enum import Enum, unique

from sensetrack.ot import algoritmi, interfaces

@unique
class AlgFactory(Enum):
    """Factory class for creating algorithm instances."""
    OPENCVOF = algoritmi.OpenCVOpticalFlow
    SKIOFILK = algoritmi.SkiOpticalFlowILK
    SKIOFTVL1 = algoritmi.SkiOpticalFlowTVL1
    SKIPCCV = algoritmi.SkiPCC_Vector


def get_method(algname: str) -> interfaces.OTAlgorithm:
    """Get an algorithm instance by name."""
    return eval(f"AlgFactory.{algname}.value")
