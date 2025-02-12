from enum import Enum, unique

from ot import algoritmi, interfaces


@unique
class AlgFactory(Enum):
    OPENCVOF = algoritmi.OpenCVOpticalFlow
    SKIOFILK = algoritmi.SkiOpticalFlowILK
    SKIOFTVL1 = algoritmi.SkiOpticalFlowTVL1
    SKIPCCV = algoritmi.SkiPCC_Vector


def get_algorithm(algname: str) -> interfaces.OTAlgorithm:
    return eval(f"AlgFactory.{algname}.value")
