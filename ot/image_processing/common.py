"""
This module provides various functions for normalizing and transforming 2D arrays and image data. 
It includes utilities for applying rolling windows, converting arrays to 8-bit unsigned integers, 
equalizing histograms of image channels, and performing different types of normalization such as 
standard normalization, power transformation, and z-score normalization. Additionally, it offers 
functions to handle raster data using the rasterio library.
"""
import cv2
import numpy as np
from skimage import color

from log import setup_logger
from ot.interfaces import Image

logger = setup_logger(__name__)

DEFAULT_NODATA = np.nan


def _np_to_cv2_dtype(array: np.ndarray) -> int:
    dtype_map = dict(uint8=cv2.CV_8U,
                     int8=cv2.CV_8S,
                     uint16=cv2.CV_16U,
                     int16=cv2.CV_16S,
                     int32=cv2.CV_32S,
                     float32=cv2.CV_32F,
                     float64=cv2.CV_64F)
    return dtype_map.get(array.dtype.name, None)


def _get_cv2_norm_name(norm: int) -> str:
    norm_map = {
        cv2.NORM_INF: "NORM_INF",
        cv2.NORM_L1: "NORM_L1",
        cv2.NORM_L2: "NORM_L2",
        cv2.NORM_L2SQR: "NORM_L2SQR",
        cv2.NORM_HAMMING: "NORM_HAMMING",
        cv2.NORM_HAMMING2: "NORM_HAMMING2",
        cv2.NORM_TYPE_MASK: "NORM_TYPE_MASK",
        cv2.NORM_RELATIVE: "NORM_RELATIVE",
        cv2.NORM_MINMAX: "NORM_MINMAX"
    }
    return norm_map.get(norm, None)


def _cv2_to_np_dtype(dtype: int) -> str:
    dtype_map = {
        cv2.CV_8U: str(np.uint8),
        cv2.CV_8S: str(np.int8),
        cv2.CV_16U: str(np.uint16),
        cv2.CV_16S: str(np.int16),
        cv2.CV_32S: str(np.int32),
        cv2.CV_32F: str(np.float32),
        cv2.CV_64F: str(np.float64)
    }
    return dtype_map.get(dtype, None)


def _array_verbose(array: np.ndarray) -> np.ndarray:
    logger.debug(f"{array.shape=} {array.dtype=}")
    logger.debug(f"min = {float(array.min())}; max = {float(array.max())}")
    return array


def _tofromimage(func, **kwargs):
    def inner(**kwargs):
        affine = kwargs["array"].affine
        crs = kwargs["array"].crs
        kwargs["array"] = kwargs["array"].image

        altered_array = func(**kwargs)
        return Image(altered_array, affine, crs)

    return inner


def _is_cv8u(array: np.ndarray) -> bool:
    return _np_to_cv2_dtype(array) == cv2.CV_8U


def _is_multiband(array) -> bool: return array.ndim > 2


def _normalize(array, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U):
    if not isinstance(array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    logger.debug(
        f"Normalizzo immagine nell'intervallo: ({alpha}, {beta})." +
        f" Normalizzazione: {_get_cv2_norm_name(cv2.NORM_MINMAX)}." +
        f" Formato in uscita: {_cv2_to_np_dtype(dtype)}")

    return cv2.normalize(array, dst=None, alpha=alpha, beta=beta,
                         norm_type=norm_type, dtype=dtype)


def _overwrite_nodata(array, nodata: int | float = -9999.) -> np.ndarray:
    logger.debug(
        f"Valori pari a {nodata} intesi come valori NULLI (sovrascritti come DEFAULT_NODATA)")
    array = array.copy()
    array[array == nodata] = DEFAULT_NODATA
    return array


def to_single_band_uint8(array, **kwargs) -> np.ndarray:

    _array_verbose(array)

    if not _is_cv8u(array):
        logger.debug("Converto dtype in CV_8U")
        array = _normalize(array)

    if _is_multiband(array):
        logger.debug("Array multibanda (ndim > 2)")
        array = _normalize(color.rgb2gray(array))

    return array


def svd_filter(patch: np.ndarray, k: int = 5, full_matrices=False):
    U, stride, VT = np.linalg.svd(patch, full_matrices=full_matrices)

    # Mantieni solo i primi k valori singolari
    S_filtered = np.zeros_like(stride)
    S_filtered[:k] = stride[:k]
    S_matrix = np.diag(S_filtered)

    # Ricostruzione della patch
    return np.dot(U, np.dot(S_matrix, VT))