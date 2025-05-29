import subprocess as sp
import sys
from pathlib import Path

from sensetrack.cosmo import lib
from sensetrack.log import setup_logger
from sensetrack.snap_gpt.lib import (GPTSubsetter, Graphs, SARPreprocessor,
                                     SARPreprocessing)

logger = setup_logger(__name__)


class CSKPreprocessor(SARPreprocessor):
    REFERENCE_RES = 6.0

    def __init__(self, SUBSET: GPTSubsetter, PROCESS: SARPreprocessing) -> None:
        super().__init__(SUBSET, SUBSET)

    def run(self, CSKFILE: str | Path) -> None:

        CSKFILE = Path(CSKFILE)

        if not CSKFILE.suffix == '.h5':
            raise ValueError("Is not a valid cosmo format")

        csk_info = lib.CSKProduct.parse_filename(CSKFILE.stem)

        OUTPUT_FILE = csk_info.OrbitDirection[0] + "_" +\
            self.SUBSET.name + "_" +\
            CSKFILE.stem + ".tif"

        sp.run(["gpt.exe",
                self.GRAPH,
                f'-Pinput='+str(CSKFILE),
                '-Psubset='+self.SUBSET.geometry.__str__(),
                f'-Prm={self.REFERENCE_RES}',
                '-Pooutput='+str(CSKFILE.parent / OUTPUT_FILE)
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
    CSKFILE = Path(sys.argv[3])

    if not CSKFILE.is_file():
        print(f"The file {CSKFILE} does not exist.")
        exit(-1)

    else:
        CSKPreprocessor(SUBSET, PROCESS).run(CSKFILE)
        exit(1)
