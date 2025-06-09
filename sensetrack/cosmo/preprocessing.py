"""
preprocessing.py
This module provides preprocessing utilities for COSMO-SkyMed (CSK) and COSMO Second Generation (CSG) Synthetic Aperture Radar (SAR) products.
It defines preprocessor classes for both CSK and CSG product types, enabling automated multi-look parameter estimation, subsetting, and execution
of SNAP GPT workflows for SAR data processing.

Classes:

    - CSKPreprocessor: Preprocessor for CSK SAR products. Handles multi-look estimation, file validation, and invokes SNAP GPT workflows.
    - CSGPreprocessor: Preprocessor for CSG SAR products. Similar to CSKPreprocessor but tailored for CSG product metadata and conventions.

Functions:

    - main(preprocessor, file, workflow, aoi): Entry point for command-line execution. Parses arguments, validates inputs, and runs the selected preprocessor.

Usage:

    This module is intended to be used as a command-line tool for preprocessing SAR data. It requires specification of the product type (CSK or CSG),
    the input file, the desired processing workflow, and the area of interest (AOI) as a shapefile or GeoPackage.

Dependencies:

    - sensetrack.cosmo.lib: Provides product metadata parsing and utilities.
    - sensetrack.log: Logging setup.
    - sensetrack.snap_gpt.lib: SNAP GPT workflow and subsetting utilities.
    - subprocess, pathlib, argparse, sys

Example:

    python preprocessing.py --product_type CSK --file /path/to/file.h5 --workflow SOME_WORKFLOW --aoi /path/to/aoi.shp

Raises:

    - ValueError: If the input file is not a valid COSMO format.
    - AssertionError: If the specified workflow is not supported.
    - FileNotFoundError: If the AOI file does not exist.
    - SystemExit: On critical errors or after successful processing.
"""
import subprocess as sp
from pathlib import Path

from sensetrack.cosmo import lib
from sensetrack.log import setup_logger
from sensetrack.snap_gpt.lib import (CSG_HIMAGE_SLC, CSK_HIMAGE_SLC,
                                     GPTSubsetter, SARPreprocessor)

logger = setup_logger(__name__)


class CosmoPreprocessor(SARPreprocessor):
    """
    A preprocessor class for COSMO-Skymed (CSG & CSK) SAR products.

    This class extends the SARPreprocessor to handle Cosmo-specific preprocessing operations,
    including multi-look estimation and SNAP GPT workflow execution.

    Args:
        SUBSET (GPTSubsetter): A subsetter object defining the area of interest.
        PROCESS (SARPreprocessing): The processing workflow to be applied.
    """
    def __init__(self, SUBSET: GPTSubsetter, PROCESS: str) -> None:
        super().__init__(SUBSET, PROCESS)

    def run(self, COSMOFILE: str | Path, CRS: str = "EPSG:32632") -> None:
        """
        Execute preprocessing workflow on a CSG SAR product.

        Args:
            CSGFILE (str | Path): Path to the CSG product file (.h5 format)
            CRS (str, optional): Coordinate Reference System. Defaults to "EPSG:32632".

        Raises:
            ValueError: If the input file is not a valid CSG format (.h5)
        """

        info = lib.CosmoFilenameParser.create_from_filename(COSMOFILE)

        if info.Mission == "CSG":
            ml = self.estimate_multilook_parms(COSMOFILE, CSG_HIMAGE_SLC(), 2)

        elif info.Mission == "CSK":
            ml = self.estimate_multilook_parms(COSMOFILE, CSK_HIMAGE_SLC(), 2)

        logger.debug(f"MultiLook parameters and final resolution: {ml}")

        COSMOFILE = Path(COSMOFILE)

        OUTPUT_FILE = f"{info.Mission}_" +\
            f"{info.OrbitDirection[0]}_" +\
            f'{info.Polarization}_' +\
            f'{info.SensingStartTime.isoformat()}_' +\
            f"[{self.SUBSET.name}]" +\
            f'[{self.PROCESS}].tif'

        sp.run(["gpt.exe",
                self.GRAPH,
                f'-Pinput='+str(COSMOFILE),
                f'-Pinput='+str(COSMOFILE),
                f'-PnRgLooks={ml.Num_Range_LOOKS}',
                f'-PnAzLooks={ml.Num_Azimuth_LOOKS}',
                f'-PmapProjection={CRS}',
                f'-PgeoRegion={self.SUBSET.geometry.__str__()}',
                f'-Poutput={str(COSMOFILE.parent / OUTPUT_FILE)}'
                f'-Poutput={str(COSMOFILE.parent / OUTPUT_FILE)}'
                ],
               shell=True)

        return COSMOFILE.parent / OUTPUT_FILE

def main(file, graph_name, aoi):
    """
    Main entry point for preprocessing COSMO-SkyMed and COSMO Second Generation
    SAR products.

    Args:
        file (str): Path to the input SAR product file
        graph_name (str): Name of the processing graph to apply
        aoi (str): Path to the Area of Interest file (ESRI Shapefile or GeoPackage)

    Raises:
        FileNotFoundError: If any of the required files do not exist.
        ValueError: (1) If filename has either a wrong extension or is not compliant
        with Cosmo-Skymed naming convention; (2) invalid AOI specifications.
        RuntimeError: For errors during external process execution.
        Exception: For any other unexpected errors.
    """

    try:
        SUBSET = GPTSubsetter.get_subset(aoi)
        preprocessor = CosmoPreprocessor(SUBSET, graph_name)
        OUTPUT_FILEPATH = preprocessor.run(Path(file))
        logger.info(f"Cosmo product {file} successfully converted in {OUTPUT_FILEPATH}")
        exit(1)

    except FileNotFoundError as err:
        logger.error(f"File not found: {err}")
        exit(0)
    except ValueError as err:
        logger.error(f"Value error: {err}")
        exit(0)
    except RuntimeError as err:
        logger.error(f"Runtime error: {err}")
        exit(0)
    except Exception as err:
        logger.error(f"Unexpected error: {err}")
        exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)

    parser.add_argument("--file", required=True,
                        help="Cosmo-Skymed product file path",
                        type=str)
    parser.add_argument("--graph_name", required=True,
                        help="Processing graph to be used",
                        help="Cosmo-Skymed product file path",
                        type=str)
    parser.add_argument("--graph_name", required=True,
                        help="Processing graph to be used",
                        type=str)
    parser.add_argument("--aoi", required=True,
                        help="Area Of Interest (ESRI Shapefile or GeoPackage)",
                        help="Area Of Interest (ESRI Shapefile or GeoPackage)",
                        type=str)

    kwargs = vars(parser.parse_args())
    kwargs = vars(parser.parse_args())

    main(**kwargs)
    main(**kwargs)