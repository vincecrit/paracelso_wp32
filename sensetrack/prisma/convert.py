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
from typing import Iterable

import cv2
import h5py
import numpy as np
import rasterio as rio
from rasterio.transform import from_bounds

from sensetrack.ot.interfaces import Image
from sensetrack.ot.lib import image_to_rasterio

hdf5_dir = "hdf5"
gtif_dir = "gtif"


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


def get_swir_bandwidths(h5_file):
    """
    Extracts the SWIR band center wavelengths and FWHM values from a PRISMA HDF5 file.

    Args:
        h5_file (str or Path): Path to the PRISMA HDF5 file.

    Returns:
        np.ndarray: An array with columns [center + FWHM/2, center - FWHM/2, FWHM] for each SWIR band.

    Raises:
        ValueError: If the required attributes are not found or are not numpy arrays.
    """
    with h5py.File(h5_file, "r") as f:
        list_cw_swir = f['/'].attrs['List_Cw_Swir']
        list_fwhm_swir = f['/'].attrs['List_Fwhm_Swir']

    if isinstance(list_fwhm_swir, np.ndarray) and isinstance(list_cw_swir, np.ndarray):
        swir_info = np.c_[list_cw_swir + (list_fwhm_swir/2),
                          list_cw_swir - (list_fwhm_swir/2),
                          list_fwhm_swir]
        return swir_info
    else:
        raise ValueError("No data found")


def get_vnir_bandwidths(h5_file):
    """
    Extracts the VNIR band center wavelengths and FWHM values from a PRISMA HDF5 file.

    Args:
        h5_file (str or Path): Path to the PRISMA HDF5 file.

    Returns:
        np.ndarray: An array with columns [center + FWHM/2, center - FWHM/2, FWHM] for each VNIR band.

    Raises:
        ValueError: If the required attributes are not found or are not numpy arrays.
    """
    with h5py.File(h5_file, "r") as f:
        list_cw_vnir = f['/'].attrs['List_Cw_Vnir']
        list_fwhm_vnir = f['/'].attrs['List_Fwhm_Vnir']

    if isinstance(list_fwhm_vnir, np.ndarray) and isinstance(list_cw_vnir, np.ndarray):
        vnir_info = np.c_[list_cw_vnir + (list_fwhm_vnir/2),
                          list_cw_vnir - (list_fwhm_vnir/2),
                          list_fwhm_vnir]
        return vnir_info
    else:
        raise ValueError("No data found")


def get_prisma_dataset_bounds(src: h5py.File) -> dict:
    """
    Extracts the bounding coordinates (west, south, east, north) from a PRISMA HDF5 file.

    Args:
        src (h5py.File): An open HDF5 file object from a PRISMA product.

    Returns:
        dict: A dictionary with keys 'west', 'south', 'east', 'north' and their corresponding float values.

    Raises:
        ValueError: If the input is not an HDF5 file.
    """
    corners = [
        "Product_LLcorner_easting",
        "Product_LRcorner_northing",
        "Product_URcorner_easting",
        "Product_ULcorner_northing"
    ]

    bound_names = ["west", "south", "east", "north"]
    bounds = dict()

    if isinstance(src, h5py.File):
        for bound, corner in zip(bound_names, corners):
            bounds[bound] = float(src.attrs.get(corner, None))

        return bounds
    else:
        raise ValueError(f"{src} is not a h5 file")


def get_band_from_h5dataset_array(ds: h5py.Dataset, band: int) -> np.ndarray:
    """
    Extracts a specific band from a 3D HDF5 dataset array.

    Args:
        ds (h5py.Dataset): The HDF5 dataset with shape (rows, bands, columns).
        band (int): The band index to extract.

    Returns:
        np.ndarray: The 2D array corresponding to the selected band.

    Raises:
        IndexError: If the requested band index is out of range.
    """
    try:
        return ds[:, band, :]
    except IndexError:
        raise IndexError(
            f"Maximum number of bands available for datacube is {ds.shape[1]}")


def get_prisma_image(prisma_file: str | Path,
                     datacube: str = 'pan',
                     band: Iterable | int | None = None) -> Image:
    """
    Extracts a specific band or bands from a PRISMA HDF5 file and returns an Image object.

    Args:
        prisma_file (str | Path): Path to the PRISMA HDF5 file.
        datacube (str): The datacube to extract from ('pan', 'swir', or 'vnir').
        band (Iterable | int): The band index (int) or list of indices (Iterable) to extract.

    Returns:
        Image: An Image object containing the extracted band(s), affine transform, CRS, and nodata value.

    Raises:
        AssertionError: If the dataset or bandwidths are not found or are of unexpected type.
        IndexError: If the requested band index is out of range.
        ValueError: If the datacube is not recognized.
    """
    with h5py.File(prisma_file) as src:
        epsg = src.attrs["Epsg_Code"]

        bounds = get_prisma_dataset_bounds(src)

        match datacube:
            case 'pan':
                ds = src.get(
                    "HDFEOS/SWATHS/PRS_L2D_PCO/Data Fields/Cube", None)
                band = None
                bandwidths = None
            case 'swir':
                ds = src.get(
                    "HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube", None)
                bandwidths = get_swir_bandwidths(prisma_file)
            case 'vnir':
                ds = src.get(
                    "HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube", None)
                bandwidths = get_vnir_bandwidths(prisma_file)

        assert isinstance(
            ds, h5py.Dataset), f"No dataset found in `{datacube}`"
        assert isinstance(bandwidths, np.ndarray)

        if band is None:
            array = ds[:, :]
            height, width = array.shape

        elif isinstance(band, int):
            array = get_band_from_h5dataset_array(ds, band)
            height, width = array.shape

        elif isinstance(band, Iterable) and not isinstance(band, str):
            channels = [get_band_from_h5dataset_array(ds, b) for b in band]
            array = cv2.merge(channels)
            height, width, _ = array.shape

        affine = from_bounds(**bounds, width=width, height=height)

        return Image(array, affine, "EPSG:"+str(epsg), 0)


def _get_parser():
    parser = ArgumentParser()
    parser.add_argument('-d', '--datacube', default='pan', type=str,
                        help='Dataset (avaiable options: `pan`, `swir` e `vnir`)')
    parser.add_argument('-b', '--band', default=0, type=int,
                        help='Selected band')
    parser.add_argument('-f', '--file', help='PRISMA file (*.h5)')

    return parser


def main():

    parms = _get_parser().parse_args()

    get_prisma_info(parms.file)
    img = get_prisma_image(parms.file, parms.datacube, parms.band)
    wd = Path(parms.file).parent
    stem = Path(parms.file).stem
    outfile = wd / (stem+f"[{parms.datacube}_{parms.band}].tif")
    image_to_rasterio(img, outfile)


if __name__ == "__main__":
    main()
