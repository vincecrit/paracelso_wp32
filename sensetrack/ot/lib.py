"""
This module provides utility functions for managing raster images and geospatial vector data,
with a focus on loading, saving, coregistration, and conversion between different formats
using the `rasterio`, `geopandas`, and `opencv` libraries.
"""

import logging
from pathlib import Path
from typing import Callable

import cv2
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.drivers
from rasterio.errors import (
    DriverCapabilityError,
    DriverRegistrationError,
    PathError,
    RasterioIOError,
)
from rasterio.warp import Resampling, calculate_default_transform, reproject

from sensetrack.log import setup_logger
from sensetrack.ot.interfaces import Image

logger = setup_logger(__name__)


def get_band(image: Image, n: str | int = 0) -> Image | None:
    """
    Get a specific band from the image.

    Args:
        n (str | int | None): Band identifier. Can be band name (str),
                                index (int), or None for entire image

    Returns:
        Image: New Image instance containing the requested band
    """
    if isinstance(n, str):
        band = image.__getattribute__(n)
        return Image(band, image.affine, image.crs, image.nodata)

    elif isinstance(n, int):
        band = image[n]
        return Image(band, image.affine, image.crs, image.nodata)

    else:
        return


def _to_bandlast(arr):
    """
    Convert array from [BAND, ROW, COL] format to [ROW, COL, BAND] format.

    Args:
        arr: Input array in [BAND, ROW, COL] format

    Returns:
        np.ndarray: Array in [ROW, COL, BAND] format
    """
    return np.transpose(arr, (1, 2, 0))


def __debug_attrerr(func, *args, **kwargs):
    """
    Decorator to handle AttributeError exceptions with logging.

    Args:
        func: Function to wrap
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        callable: Wrapped function that handles AttributeError
    """

    def inner(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except AttributeError as err:
            logger.critical(f"{err.__class__.__name__}: {err.__str__()}")
        return None

    return inner


def is_image(arg) -> bool:
    """
    Check if argument is an Image instance.

    Args:
        arg: Object to check

    Returns:
        bool: True if arg is an Image instance, False otherwise
    """
    return isinstance(arg, Image)


def is_geodf(arg) -> bool:
    """
    Check if argument is a GeoDataFrame instance.

    Args:
        arg: Object to check

    Returns:
        bool: True if arg is a GeoDataFrame instance, False otherwise
    """
    return isinstance(arg, gpd.GeoDataFrame)


def is_identity_affine(affine: rasterio.Affine) -> bool:
    """
    Check if affine transform is an identity matrix.

    Args:
        affine (rasterio.Affine): Affine transformation matrix to check

    Returns:
        bool: True if the affine transform is an identity matrix, False otherwise
    """
    matrix = np.array(affine).reshape(3, 3)
    if (np.diagonal(matrix) == 1).all():
        return True
    else:
        return False


def rasterio_open(source: str | Path, band: int | None = None) -> tuple:
    """
    Open a raster file using rasterio and convert to OpenCV format.

    Args:
        source (str): Path to the raster file
        band (int | None): Specific band to read. If None, reads up to first 3 bands

    Returns:
        tuple: (dataset, affine_transform, crs) where dataset is in OpenCV format

    Raises:
        ValueError: If requested band index is out of range
        RasterioIOError: If file cannot be opened
    """
    source = Path(source)
    logger.debug(f"Loading {source} with rasterio.")

    try:
        with rasterio.open(source) as src:
            if band is None:
                iter_bands = range(min(3, src.count))
            elif band > (src.count - 1):
                raise ValueError(
                    f"Band index {band} out of range. Dataset has {src.count} bands"
                )
            else:
                iter_bands = [band]

            channels = []
            for b in iter_bands:
                band_data = src.read(b + 1)
                channels.append(band_data)

            dataset = cv2.merge(channels)
            affine = src.meta["transform"]
            crs = src.meta["crs"]

            return dataset, affine, crs
    except RasterioIOError as e:
        logger.error(f"Failed to open {source}: {e}")
        raise


def image_to_geotiff(img: Image, outfile: str | Path) -> None:
    """
    Save an Image object to a raster file using rasterio.

    Args:
        img (Image): Image object to save
        outfile: Path to output file

    Raises:
        ValueError: If img is not an Image instance
        Various rasterio errors if file cannot be written
    """
    outfile = Path(outfile)

    if not is_image(img):
        raise ValueError(
            f"Invalid argument type. Expected {type(Image)}, got {type(img)}"
        )

    driver = rasterio.drivers.driver_from_extension(outfile)
    try:
        with rasterio.open(
            outfile,
            "w",
            transform=img.affine,
            driver=driver,
            crs=img.crs,
            nodata=img.nodata,
            width=img.width,
            height=img.height,
            dtype=img.image.dtype,
            count=1,
        ) as ds:
            ds.write(img.image, 1)
    except (
        DriverCapabilityError,
        DriverRegistrationError,
        PathError,
        RasterioIOError,
    ) as err:
        logging.critical(f"[RASTERIO] {err.__class__.__name__}: {err}")
        exit(0)


def geopandas_to_ogr(frame, outfile) -> None:
    """
    Save a GeoDataFrame to a vector file format.

    Args:
        frame: GeoDataFrame to save
        outfile: Path to output file

    Raises:
        ValueError: If frame is not a GeoDataFrame instance
    """
    if not is_geodf(frame):
        raise ValueError(
            f"Invalid argument type. Expected {type(gpd.GeoDataFrame)}, got {type(frame)}"
        )

    frame.to_file(outfile, layer="displacements")


def write_output(output, outfile: str | Path) -> None:
    """
    Write output to file in appropriate format based on file extension.

    Args:
        output: Data to write (Image or GeoDataFrame)
        outfile (str | Path): Path to output file

    Raises:
        NotImplementedError: If file extension is not supported
    """
    outfile = Path(outfile)

    match outfile.suffix:
        case ".tiff":
            image_to_geotiff(output, outfile)
        case ".tif":
            image_to_geotiff(output, outfile)
        case ".jpg":
            image_to_geotiff(output, outfile)
        case ".jpeg":
            image_to_geotiff(output, outfile)
        case ".png":
            image_to_geotiff(output, outfile)
        case ".gpkg":
            geopandas_to_ogr(output, outfile)
        case ".shp":
            geopandas_to_ogr(output, outfile)
        case _:
            raise NotImplementedError(f"Unsupported file extension: {outfile.suffix}")


def load_images(*args, nodata: float, **kwargs):
    """
    Load a pair of images and return them as Image objects.

    This function performs several steps:
    1. File extension validation
    2. Georeferencing check
    3. Image coregistration
    4. Output as Image objects

    Args:
        *args (str): Paths to reference and target images, in that order
        nodata: Value to use for nodata pixels
        **kwargs: Additional arguments passed to rasterio_open

    Returns:
        tuple[Image, Image]: Reference and target images as Image objects

    Raises:
        SystemExit: If images have different formats or inconsistent georeferencing
    """
    #  [1] Create Path objects
    reference_file, target_file = [Path(src) for src in args]
    logger.info(f"REFERENCE: {reference_file.name}")
    logger.info(f"TARGET: {target_file.name}")

    # Files with different extensions are not accepted
    if not reference_file.suffix == target_file.suffix:
        logger.critical("Images have different formats")
        logger.debug(f"{reference_file.suffix=}, {target_file.suffix=}")
        exit(0)

    else:
        # Load any file type with rasterio
        reference = Image(*rasterio_open(reference_file, **kwargs), nodata=nodata)
        target = Image(*rasterio_open(target_file, **kwargs), nodata=nodata)

        # [2] rasterio assigns an identity Affine object when
        # georeferencing is not defined
        are_identity_affines = [
            is_identity_affine(e.affine) for e in (reference, target)
        ]

        # If neither file is georeferenced, should be fine
        if all(are_identity_affines):
            logger.warning(
                "Neither file has georeferencing. "
                + "Coregistration will be limited to aligning/scaling target image pixels"
            )
            pass

        # If georeferencing is defined for only one file,
        # we don't know what to do... so exit
        elif any(are_identity_affines):
            logger.critical("One of the files lacks georeferencing.")
            logger.debug(f"{is_identity_affine(reference.affine)=}")
            logger.debug(f"{is_identity_affine(target.affine)=}")
            exit(0)

        # [3] Image coregistration
        else:
            if not reference.is_coregistered(target):
                logger.info("Performing raster image coregistration")
                target = basic_pixel_coregistration(
                    str(target_file), str(reference_file)
                )
            else:
                # This never actually happens
                logger.info("Raster images already coregistered.")

        return reference, target


def _load_as_image(func, *args, **kwargs) -> Callable:
    def inner(*args, **kwargs):
        output_file = func(*args, **kwargs)
        logger.info(
            f"Coregistration completed successfully. "
            + f"Coregistered file: {output_file}"
        )
        return Image(*rasterio_open(output_file))

    return inner


@_load_as_image
def basic_pixel_coregistration(
    source: str, match: str, outfile: str | Path | None = None
):
    """
    Align pixels between a target image and a reference image.

    This function performs pixel-level alignment and optionally reprojects
    to the same CRS as the reference image.

    Args:
        infile (str): Path to target image to be aligned
        match (str): Path to reference image
        outfile (str | None): Path for output file. If None, appends '_coreg' to input filename

    Returns:
        Path: Path to coregistered output file

    The function:
    1. Calculates output affine transform
    2. Sets up output metadata
    3. Reprojects each band using bilinear resampling
    """
    if outfile is None:
        out_stem = Path(source).stem + "_coreg" + Path(source).suffix
        outfile = Path(source).parent / out_stem
    else:
        outfile = Path(outfile)

    with rasterio.open(source) as src:
        src_transform = src.transform
        nodata = src.meta["nodata"]

        with rasterio.open(match) as mtc:
            dst_crs = mtc.crs

            # Calculate output affine transform
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs,
                dst_crs,
                mtc.width,
                mtc.height,
                *mtc.bounds,  # (left, bottom, right, top)
            )

        # Set output metadata
        dst_kwargs = src.meta.copy()
        dst_kwargs.update(
            {
                "crs": dst_crs,
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height,
                "nodata": nodata,
            }
        )

        logger.debug(f"Coregistered image dimensions: {dst_height}, {dst_width}")

        # Output
        logger.info(f"Exporting image: {outfile}")
        with rasterio.open(outfile, "w", **dst_kwargs) as dst:
            # Iterate through all bands in 'infile'
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )

        return outfile
