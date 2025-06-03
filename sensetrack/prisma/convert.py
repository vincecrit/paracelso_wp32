"""
convert.py
This module provides utilities for handling PRISMA satellite data files,
specifically for extracting metadata and converting data cubes to
GeoTIFF format.

Functions:

    - get_prisma_info(prisma_file): Prints selected metadata attributes
    from a PRISMA HDF5 file.
    - prisma_panchromatic_to_gtiff(prisma_file, band='swir'): Converts
    a specified band ('pan', 'swir', or 'vnir') from a PRISMA HDF5 file
    to a GeoTIFF file, preserving georeferencing information.

Command-line interface:

    - Use the '-i' or '--show_info' flag to display metadata information
    from a PRISMA file.
    - Use the '-c' or '--convert' flag to convert a PRISMA file to GeoTIFF
    format.
    - Use the '-f' or '--file' option to specify the input PRISMA file.

Dependencies:

    - h5py
    - numpy
    - rasterio

Note:

    This script is intended for use with PRISMA satellite data products in HDF5 format.
"""
from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
import rasterio as rio

hdf5_dir = "hdf5"
gtif_dir = "gtif"


def _get_parser():
    parser = ArgumentParser()
    parser.add_argument('-i', '--show_info', action='store_true',
                        default=False, help='mostra alcune informazioni')
    parser.add_argument('-c', '--convert', action='store_true',
                        default=True, help='converte in geotiff')
    parser.add_argument('-d', '--datacube', default='pan', type=str,
                        help='Dataset da esportare (opzioni possibili: `pan`, `swir` e `vnir`)')
    parser.add_argument('-f', '--file', help='file prisma')

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


def prisma_panchromatic_to_gtiff(prisma_file: str | Path, datacube='swir'):
    prisma_file = Path(prisma_file)

    valid_cubes = ['pan', 'swir', 'vnir']
    if datacube not in valid_cubes:
        raise ValueError(
            f"Invalid band '{datacube}'. Valid options are {valid_cubes}.")

    with h5py.File(prisma_file) as src:
        epsg = src.attrs["Epsg_Code"]

        west = min(src.attrs["Product_LLcorner_easting"],
                   src.attrs["Product_ULcorner_easting"])
        south = min(src.attrs["Product_LLcorner_northing"],
                    src.attrs["Product_LRcorner_northing"])
        east = max(src.attrs["Product_LRcorner_easting"],
                   src.attrs["Product_URcorner_easting"])
        north = max(src.attrs["Product_ULcorner_northing"],
                    src.attrs["Product_URcorner_northing"])

        match datacube:
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
        "fp": prisma_file.parent / (prisma_file.stem + f"_{datacube}.tif"),
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
        prisma_panchromatic_to_gtiff(parms.file, datacube = parms.datacube)

    elif parms.show_info:
        get_prisma_info(parms.file)
