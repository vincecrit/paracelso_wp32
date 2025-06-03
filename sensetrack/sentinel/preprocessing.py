"""
sensetrack.sentinel.preprocessing
=================================
This module provides preprocessing utilities for Sentinel-1 SAR data using SNAP GPT workflows.
It defines the `S1Preprocessor` class, which orchestrates the application of SNAP GPT graphs
to Sentinel-1 SLC products, including subsetting and SAR-specific preprocessing steps.
Classes
-------
- S1Preprocessor: Inherits from `SARPreprocessor` and implements the preprocessing pipeline
    for Sentinel-1 data, including multilook estimation, orbit property extraction, and
    execution of SNAP GPT workflows.
Functions
---------
- main(preprocessor, file, workflow, aoi): Entry point for command-line execution. Parses
    arguments, validates inputs, and runs the specified preprocessing workflow.
Usage
-----
This module is intended to be run as a script with command-line arguments specifying the
input SAR file, the desired processing workflow, and the area of interest (AOI) as a
shapefile or GeoPackage.
Example:
        python preprocessing.py --file <SARFILE.zip> --workflow <WORKFLOW> --aoi <AOI_FILE>
Dependencies
------------
- sensetrack.log
- sensetrack.sentinel.utils
- sensetrack.snap_gpt.lib
- SNAP GPT (gpt.exe must be available in the system path)
Raises
------
- ValueError: If the input SAR file is not a zip archive.
- AssertionError: If the specified workflow is not supported.
- FileNotFoundError, ValueError: If the AOI file is invalid or not found.
Logging
-------
The module uses a logger to provide debug and error messages during processing.
"""
import subprocess as sp
from pathlib import Path

from sensetrack.log import setup_logger
from sensetrack.sentinel.utils import read_orbit_properties
from sensetrack.snap_gpt.lib import (S1_IW_SLC, GPTSubsetter, SARPreprocessing,
                                     SARPreprocessor)

logger = setup_logger(__name__)


class S1Preprocessor(SARPreprocessor):
    def __init__(self, SUBSET: GPTSubsetter, PROCESS: SARPreprocessing) -> None:
        super().__init__(SUBSET, PROCESS)

    def run(self, SARFILE: str | Path, CRS: str = "EPSG:32632") -> None:

        ml = self.estimate_multilook_parms(SARFILE, S1_IW_SLC(), 1)
        # res = int(min(ml.Estimated_AzimuthResolution, ml.Estimated_RangeResolution))
        logger.debug(f"Parametri MultiLook e risoluzione finale: {ml}")
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
                f'-Pinput={SARFILE}',
                f'-PnRgLooks={ml.Num_Range_LOOKS}',
                f'-PnAzLooks={ml.Num_Azimuth_LOOKS}',
                f'-PgeoRegion={self.SUBSET.geometry.__str__()}',
                f'-Poutput={str(SARFILE.parent / OUTPUT_FILE)}',
                f'-PmapProjection={CRS}'
                ],
               shell=True)


def main(preprocessor, file, workflow, aoi):

    import sys

    __processes = list(SARPreprocessing._member_map_.keys())

    assert workflow.upper() in __processes

    try:
        SUBSET = GPTSubsetter.get_subset(aoi)

    except (FileNotFoundError, ValueError) as err:
        logger.error(f"Failed to get AOI: {err}")
        exit(-1)

    PROCESS = eval(f"SARPreprocessing.{sys.argv[2].upper()}")
    SARFILE = Path(file)

    if not SARFILE.is_file():
        print(f"The file {SARFILE} does not exist.")
        exit(-1)

    else:
        preprocessor(SUBSET, PROCESS).run(SARFILE)
        exit(1)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True,
                        help="Percorso al file da elaborare", type=str)
    parser.add_argument("--workflow", required=True,
                        help="Workflow da utilizzare per il processamento",
                        type=str)
    parser.add_argument("--aoi", required=True,
                        help="Area di interesse (ESRI Shapefile o GeoPackage)",
                        type=str)

    args = vars(parser.parse_args())

    main(S1Preprocessor, **args)


# if __name__ == "__main__":

#     import sys

#     __processes = list(SARPreprocessing._member_map_.keys())

#     assert sys.argv[2].upper() in __processes

#     try:
#         SUBSET = GPTSubsetter.get_subset(sys.argv[1])

#     except (FileNotFoundError, ValueError) as err:
#         logger.error(f"Failed to get AOI: {err}")
#         exit(-1)

#     PROCESS = eval(f"SARPreprocessing.{sys.argv[2].upper()}")
#     SARFILE = Path(sys.argv[3])

#     if not SARFILE.is_file():
#         print(f"The file {SARFILE} does not exist.")
#         exit(-1)

#     else:
#         S1Preprocessor(SUBSET, PROCESS).run(SARFILE)
#         exit(1)
