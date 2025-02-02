from enum import Enum, unique

from ot import algoritmi, interfaces


@unique
class AlgFactory(Enum):
    OPENCVOF = algoritmi.OpenCVOpticalFlow
    SKIOFILV = algoritmi.SkiOpticalFlowILV
    SKIOFTVL1 = algoritmi.SkiOpticalFlowTVL1


def get_algorithm(algname: str) -> interfaces.OTAlgorithm:
    return eval(f"AlgFactory.{algname}.value")
