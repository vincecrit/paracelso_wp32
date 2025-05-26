import subprocess as sp
import sys
from pathlib import Path

from log import setup_logger
from s1.manifest_file import read_orbit_properties
from snap_gpt.config import OUTFOLDER
from snap_gpt.lib import AOI, GPTSubsetter, Graphs, SARPreprocessing

logger = setup_logger(__name__)


class S1Preprocessor:
    def __init__(self, AOI_SUBSET: str | AOI, PROCESS: SARPreprocessing) -> None:
        if not isinstance(PROCESS, SARPreprocessing):
            raise TypeError(
                f"PROCESS must be SARPreprocessing enum, got {type(PROCESS)}")

        self._PROCESS = PROCESS.value
        try:
            self.SUBSET = GPTSubsetter.get_subset(AOI_SUBSET)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Failed to get AOI: {e}")
            raise

        graph_path = Graphs._member_map_[PROCESS.value].value
        if not graph_path.is_file():
            raise FileNotFoundError(f"Graph file not found: {graph_path}")
        self.GRAPH = graph_path

    def run(self, SARFILE: str | Path) -> None:
        SARFILE = Path(SARFILE)
        if not SARFILE.is_file():
            raise FileNotFoundError(f"SAR file not found: {SARFILE}")

        if not SARFILE.suffix == '.zip':
            raise ValueError("SAR file must be a zip archive")

        orbit_properties = read_orbit_properties(SARFILE)

        ORBIT_TAG = orbit_properties.ORBIT_PASS[0]

        OUTPUT_FILE = f'{ORBIT_TAG}_' +\
            f'{self.SUBSET.name}_' +\
            f'{self._PROCESS}_' +\
            f'{orbit_properties.RELORBIT:>03}_' +\
            f'{orbit_properties.NODE_TIME}.tif'

        if not OUTFOLDER.is_dir():
            OUTFOLDER.mkdir(parents=True)

        sp.run(["gpt.exe",
                self.GRAPH,
                f'-Pf='+str(SARFILE),
                '-Psubset='+self.SUBSET.geometry.__str__(),
                '-Po='+str(OUTFOLDER / OUTPUT_FILE)
                ],
               shell=True)


if __name__ == "__main__":
    import sys

    __sites = list(AOI._member_map_.keys())
    __processes = list(SARPreprocessing._member_map_.keys())

    if sys.argv[1].upper() in __sites:
        SUBSET = eval(f"AOI.{sys.argv[1].upper()}")
    else:
        SUBSET = sys.argv[1]

    assert sys.argv[2].upper() in __processes

    PROCESS = eval(f"SARPreprocessing.{sys.argv[2].upper()}")
    SARFILE = Path(sys.argv[3])

    if not SARFILE.is_file():
        print(f"Il file {SARFILE} non esiste.")
        exit(-1)

    else:
        S1Preprocessor(SUBSET, PROCESS).run(SARFILE)
        exit(1)
