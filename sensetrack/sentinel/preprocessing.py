import subprocess as sp
import sys
from pathlib import Path

from sensetrack.log import setup_logger
from sensetrack.sentinel.manifest_file import read_orbit_properties
from sensetrack.snap_gpt.lib import GPTSubsetter, Graphs, SARPreprocessing

logger = setup_logger(__name__)


class S1Preprocessor:
    def __init__(self, SUBSET: GPTSubsetter, PROCESS: SARPreprocessing) -> None:
        if not isinstance(PROCESS, SARPreprocessing):
            raise TypeError(
                f"PROCESS must be SARPreprocessing enum, got {type(PROCESS)}")

        self._PROCESS = PROCESS.value
        self.SUBSET = SUBSET
        GRAPH_PATH = Graphs._member_map_[PROCESS.value].value

        if not GRAPH_PATH.is_file():
            raise FileNotFoundError(f"Graph file not found: {GRAPH_PATH}")

        self.GRAPH = GRAPH_PATH

    def run(self, SARFILE: str | Path) -> None:

        SARFILE = Path(SARFILE)

        if not SARFILE.suffix == '.zip':
            raise ValueError("SAR file must be a zip archive")

        orbit_properties = read_orbit_properties(SARFILE)

        ORBIT_TAG = orbit_properties.ORBIT_PASS[0]

        OUTPUT_FILE = f'{ORBIT_TAG}_' +\
            f'{self.SUBSET.name}_' +\
            f'{self._PROCESS}_' +\
            f'{orbit_properties.RELORBIT:>03}_' +\
            f'{orbit_properties.NODE_TIME}.tif'

        sp.run(["gpt.exe",
                self.GRAPH,
                f'-Pf='+str(SARFILE),
                '-Psubset='+self.SUBSET.geometry.__str__(),
                '-Po='+str(SARFILE.parent / OUTPUT_FILE)
                ],
               shell=True)


if __name__ == "__main__":

    import sys

    __processes = list(SARPreprocessing._member_map_.keys())

    assert sys.argv[2].upper() in __processes

    try:
        SUBSET = GPTSubsetter.get_subset(sys.argv[1])

    except (FileNotFoundError, ValueError) as err:
        logger.error(f"Failed to get AOI: {err}")
        exit(-1)

    PROCESS = eval(f"SARPreprocessing.{sys.argv[2].upper()}")
    SARFILE = Path(sys.argv[3])

    if not SARFILE.is_file():
        print(f"The file {SARFILE} does not exist.")
        exit(-1)

    else:
        S1Preprocessor(SUBSET, PROCESS).run(SARFILE)
        exit(1)
