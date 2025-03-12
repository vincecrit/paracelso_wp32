"""
This module provides various functions for normalizing and transforming 2D arrays and image data. 
It includes utilities for applying rolling windows, converting arrays to 8-bit unsigned integers, 
equalizing histograms of image channels, and performing different types of normalization such as 
standard normalization, power transformation, and z-score normalization. Additionally, it offers 
functions to handle raster data using the rasterio library.
"""
import logging

import cv2
import numpy as np
from skimage import color

from ot.interfaces import Image

logger = logging.getLogger(__name__)
DEFAULT_NODATA = 0


def _array_verbose(array: np.ndarray) -> np.ndarray:
    logger.debug(f"{array.shape=} {array.dtype=}")
    logger.debug(f"{array.min()=} {array.max()=}")
    return array


def _tofromimage(func, **kwargs):
    def inner(**kwargs):
        affine = kwargs['array'].affine
        crs = kwargs['array'].crs
        kwargs['array'] = kwargs['array'].image

        altered_array = func(**kwargs)
        return Image(altered_array, affine, crs)

    return inner


def _np_to_cv2_dtype(array: np.ndarray) -> int:
    dtype_map = {
        'uint8': cv2.CV_8U,
        'int8': cv2.CV_8S,
        'uint16': cv2.CV_16U,
        'int16': cv2.CV_16S,
        'int32': cv2.CV_32S,
        'float32': cv2.CV_32F,
        'float64': cv2.CV_64F
    }
    return dtype_map.get(array.dtype.name, None)


def _is_cv8u(array) -> bool:
    return _np_to_cv2_dtype(array) == cv2.CV_8U


def _is_multiband(array) -> bool: return array.ndim > 2


def _normalize(array, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U):
    logger.debug(
        f"Normalizzo immagine in range: ({alpha},{beta})." +
        f" Normalizzazione: {cv2.NORM_MINMAX=}." +
        f" Formato in uscita: {dtype}")

    return cv2.normalize(array, dst=None, alpha=alpha, beta=beta,
                         norm_type=norm_type, dtype=dtype)


def _toCV8U(array: np.ndarray) -> np.ndarray:
    """
    Convert an array to 8-bit unsigned integer using OpenCV normalization.
    """
    return _normalize(array, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def _get_nodata_mask(array, nodata: int | float = -9999.) -> np.ndarray:
    logger.debug(
        f"Valori pari a {nodata} intesi come valori NULLI")
    return (array == nodata).any(axis=-1)


def _overwrite_nodata(array, nodata: int | float = -9999.) -> np.ndarray:
    logger.debug(
        f"Valori pari a {nodata} intesi come valori NULLI (sovrascritti come DEFAULT_NODATA)")
    array = array.copy()
    array[array == nodata] = DEFAULT_NODATA
    return array


def to_single_band_uint8(array, **kwargs) -> np.ndarray:

    _array_verbose(array)
    array = _overwrite_nodata(array, **kwargs)

    if not _is_cv8u(array):
        logger.debug("Converto dtype in CV_8U")
        array = _normalize(array)

    if _is_multiband(array):
        logger.debug("Array multibanda (ndim > 2)")
        array = _normalize(color.rgb2gray(array))

    return array
