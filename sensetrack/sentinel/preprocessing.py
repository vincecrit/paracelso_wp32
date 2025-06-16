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
from sensetrack.sentinel.lib import BadZipFile, S1ManifestParser
from sensetrack.snap_gpt.lib import (S1_IW_SLC, GPTSubsetter, Subset, SARPreprocessing,
                                     SARPreprocessor)

logger = setup_logger(__name__)


class S1Preprocessor(SARPreprocessor):
    """
    Preprocessor for Sentinel-1 SAR data using SNAP GPT.

    This class extends SARPreprocessor to implement specific preprocessing workflows
    for Sentinel-1 data, including multilook estimation, orbit property extraction,
    and SNAP GPT graph execution.
    """
    """
    Preprocessor for Sentinel-1 SAR data using SNAP GPT.
    
    This class extends SARPreprocessor to implement specific preprocessing workflows
    for Sentinel-1 data, including multilook estimation, orbit property extraction,
    and SNAP GPT graph execution.
    """

    def __init__(self, SUBSET: Subset, PROCESS: str) -> None:
        """
        Initialize the S1Preprocessor.

        Args:
            SUBSET (GPTSubsetter): Subsetter object defining the area of interest
            PROCESS (SARPreprocessing): Preprocessing workflow to apply
        """
        super().__init__(SUBSET, PROCESS)

    def run(self, S1FILE: str | Path, CRS: str = "EPSG:32632") -> Path:
        """
        Execute the preprocessing workflow on a Sentinel-1 SAR file.

        Args:
            SARFILE (str | Path): Path to input Sentinel-1 SAR file
            CRS (str, optional): Target coordinate reference system. Defaults to "EPSG:32632"

        Raises:
            ValueError: If the input file is not a zip archive
        """
        S1FILE = Path(S1FILE)

        info = S1ManifestParser.create_from_filename(S1FILE)
        ml = self.estimate_multilook_parms(str(S1FILE), S1_IW_SLC(), 1)
        logger.debug(f"Multilook parameters and final resolution: {ml}")

        ORBIT_TAG = info.ORBIT_PASS[0]

        OUTPUT_FILE = f'{ORBIT_TAG}_' +\
            f'{self.SUBSET.name}_' +\
            f'{info.RELATIVE_ORBIT:>03}_' +\
            f'{info.NODE_TIME}_' +\
            f'{self.PROCESS}.tif'

        cmd = ["gpt.exe",
               str(self.GRAPH),
               f'-Pinput={S1FILE}',
               f'-PnRgLooks={ml.Num_Range_LOOKS}',
               f'-PnAzLooks={ml.Num_Azimuth_LOOKS}',
               f'-PgeoRegion={self.SUBSET.geometry.__str__()}',
               f'-Poutput={str(S1FILE.parent / OUTPUT_FILE)}',
               f'-PmapProjection={CRS}']
        
        try:
            sp.run(cmd, check=True, capture_output=True, text=True)
        
        except sp.CalledProcessError as e:
            raise RuntimeError(f"GPT processing failed: {e.stderr}")

        logger.debug(f"SNAP Command: {"\n".join(cmd)}")

        return S1FILE.parent / OUTPUT_FILE


def main(file, graph_name, aoi):
    """
    Main entry point for SAR preprocessing execution.

    This function validates inputs, sets up the preprocessing environment,
    and executes the specified workflow.

    Args:
        file (str): Path to input SAR file
        graph_name (str): Name of preprocessing workflow to apply
        aoi (str): Path to area of interest file (shapefile or GeoPackage)

    Returns:
        None. Exits with status code:
        - -1: If AOI file is invalid or not found
        - 1: On successful completion

    Raises:
        FileNotFoundError
        ValueError
        BadZipFile
        Exception (unexpected)
    """

    try:
        SUBSET = GPTSubsetter.get_subset(aoi)
        preprocessor = S1Preprocessor(SUBSET, graph_name)
        OUTPUT_FILEPATH = preprocessor.run(Path(file))
        logger.info(
            f"S1 product {file} successfully converted in {OUTPUT_FILEPATH}")
        exit(1)

    except FileNotFoundError as err:
        logger.error(f"File not found: {err}")
        exit(0)
    except ValueError as err:
        logger.error(f"Value error: {err}")
        exit(0)
    except BadZipFile as err:
        logger.error(f"BadZipFile error: {err}")
        exit(0)
    except Exception as err:
        logger.error(f"Unexpected error: {err}")
        exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)

    parser.add_argument("--file", required=True,
                        help="Path to the SAR file to process", type=str)
    parser.add_argument("--workflow", required=True,
                        help="Preprocessing workflow to apply",
                        type=str)
    parser.add_argument("--graph_name", required=True,
                        help="Preprocessing workflow to apply",
                        type=str)
    parser.add_argument("--aoi", required=True,
                        help="Area of interest (ESRI Shapefile or GeoPackage)",
                        type=str)

    kwargs = vars(parser.parse_args())

    main(**kwargs)
