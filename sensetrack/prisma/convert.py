from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
import rasterio as rio

hdf5_dir = "hdf5"
gtif_dir = "gtif"


def _get_parser():
    parser = ArgumentParser()
    parser.add_argument('-i', '--show_info', action='store_true', default=False, help='mostra alcune informazioni')
    parser.add_argument('-c', '--convert', action='store_true', default=True, help='converte in geotiff')
    # parser.add_argument('-w', '--folder', action='store_true', default=False, help = 'converte in geotiff tutti i file in una cartella')
    parser.add_argument('-f', '--file', help = 'file prisma')

    return parser


def get_prisma_info(prisma_file: str | Path):
    prisma_file = Path(prisma_file)

    keys = [
        "Product_ID",
        "Processing_Level",
        "Cloudy_pixels_percentage",
        "Epsg_Code",
    ]

    with h5py.File(prisma_file) as src:
        for key in keys:
            print(f"{key}: {src.attrs[key]}")



def prisma_panchromatic_to_gtiff(prisma_file: str | Path, band='swir'):
    prisma_file = Path(prisma_file)

    valid_bands = ['pan', 'swir', 'vnir']
    if band not in valid_bands:
        raise ValueError(f"Invalid band '{band}'. Valid options are {valid_bands}.")

    with h5py.File(prisma_file) as src:
        epsg = src.attrs["Epsg_Code"]

        west = min(src.attrs["Product_LLcorner_easting"], src.attrs["Product_ULcorner_easting"])
        south = min(src.attrs["Product_LLcorner_northing"], src.attrs["Product_LRcorner_northing"])
        east = max(src.attrs["Product_LRcorner_easting"], src.attrs["Product_URcorner_easting"])
        north = max(src.attrs["Product_ULcorner_northing"], src.attrs["Product_URcorner_northing"])

        match band:
            case 'pan':
                array = src["HDFEOS/SWATHS/PRS_L2D_PCO/Data Fields/Cube"][()]
            case 'swir':
                array = src["HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube"][()]
            case 'vnir':
                array = src["HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube"][()]


    if array.ndim < 3:
        array = array[np.newaxis, :, :]

    count, height, width = array.shape

    profile = {
        "fp": prisma_file.parent / (prisma_file.stem + f"_{band}.tif"),
        "mode": "w",
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": count,
        "crs": rio.crs.CRS.from_epsg(epsg),
        "transform": rio.transform.from_bounds(west, south, east, north, width, height),
        "dtype": array.dtype,
    }

    with rio.open(**profile) as dst:
        for i in range(count):
            dst.write(array[i], i + 1)


if __name__ == "__main__":

    parms = _get_parser().parse_args()

    if parms.convert:
        prisma_panchromatic_to_gtiff(parms.file)

    if parms.show_info:
        get_prisma_info(parms.file)