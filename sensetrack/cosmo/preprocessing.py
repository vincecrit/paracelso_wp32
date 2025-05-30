import subprocess as sp
from pathlib import Path

from sensetrack.cosmo import lib
from sensetrack.log import setup_logger
from sensetrack.snap_gpt.lib import (CSG_HIMAGE_SLC, CSK_HIMAGE_SLC,
                                     GPTSubsetter, SARPreprocessing,
                                     SARPreprocessor)

logger = setup_logger(__name__)


class CSKPreprocessor(SARPreprocessor):
    def __init__(self, SUBSET: GPTSubsetter, PROCESS: SARPreprocessing) -> None:
        super().__init__(SUBSET, PROCESS)

    def run(self, CSKFILE: str | Path, CRS: str = "EPSG:32632") -> None:

        ml = self.estimate_multilook_parms(CSKFILE, CSK_HIMAGE_SLC(), 2)
        # res = int(min(ml.Estimated_AzimuthResolution, ml.Estimated_RangeResolution))
        logger.debug(f"Parametri MultiLook e risoluzione finale: {ml}")

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
                f'-PnRgLooks={ml.Num_Range_LOOKS}',
                f'-PnAzLooks={ml.Num_Azimuth_LOOKS}',
                f'-PmapProjection={CRS}',
                f'-PgeoRegion={self.SUBSET.geometry.__str__()}',
                f'-Poutput={str(CSKFILE.parent / OUTPUT_FILE)}'
                ],
               shell=True)


class CSGPreprocessor(SARPreprocessor):
    def __init__(self, SUBSET: GPTSubsetter, PROCESS: SARPreprocessing) -> None:
        super().__init__(SUBSET, PROCESS)

    def run(self, CSGFILE: str | Path, CRS: str = "EPSG:32632") -> None:

        ml = self.estimate_multilook_parms(CSGFILE, CSG_HIMAGE_SLC(), 2)
        # res = int(min(ml.Estimated_AzimuthResolution, ml.Estimated_RangeResolution))
        logger.debug(f"Parametri MultiLook e risoluzione finale: {ml}")

        CSGFILE = Path(CSGFILE)

        if not CSGFILE.suffix == '.h5':
            raise ValueError("Is not a valid cosmo format")

        csk_info = lib.CSGProduct.parse_filename(CSGFILE.stem)

        OUTPUT_FILE = csk_info.OrbitDirection[0] + "_" +\
            self.SUBSET.name + "_" +\
            CSGFILE.stem + ".tif"

        sp.run(["gpt.exe",
                self.GRAPH,
                f'-Pinput='+str(CSGFILE),
                f'-PnRgLooks={ml.Num_Range_LOOKS}',
                f'-PnAzLooks={ml.Num_Azimuth_LOOKS}',
                f'-PmapProjection={CRS}',
                f'-PgeoRegion={self.SUBSET.geometry.__str__()}',
                f'-Poutput={str(CSGFILE.parent / OUTPUT_FILE)}'
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
    parser.add_argument("--product_type", required=True,
                        help="Prodotto CSK o CSG", type=str)
    parser.add_argument("--file", required=True,
                        help="Percorso al file da elaborare", type=str)
    parser.add_argument("--workflow", required=True,
                        help="Workflow da utilizzare per il processamento",
                        type=str)
    parser.add_argument("--aoi", required=True,
                        help="Area di interesse (ESRI Shapefile o GeoPackage)",
                        type=str)

    args = vars(parser.parse_args())

    if args['product_type'].upper() == "CSK":
        preprocessor = CSKPreprocessor

    elif args['product_type'].upper() == "CSG":
        preprocessor = CSGPreprocessor
    
    else:
        print("Definire un prodotto Cosmo-Skymed (CSK o CSG)")
        exit(-1)

    main(preprocessor, **args)