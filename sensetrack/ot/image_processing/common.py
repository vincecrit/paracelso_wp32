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

from sensetrack.log import setup_logger
from sensetrack.ot.interfaces import Image

logger = setup_logger(__name__)

DEFAULT_NODATA = np.nan


def _np_to_cv2_dtype(array: np.ndarray) -> int:
    """
    Convert NumPy dtype to OpenCV dtype.
    
    Args:
        array (np.ndarray): Input NumPy array
        
    Returns:
        int: Corresponding OpenCV dtype constant or None if no mapping exists
    """
    dtype_map = dict(uint8=cv2.CV_8U,
                     int8=cv2.CV_8S,
                     uint16=cv2.CV_16U,
                     int16=cv2.CV_16S,
                     int32=cv2.CV_32S,
                     float32=cv2.CV_32F,
                     float64=cv2.CV_64F)
    return dtype_map.get(array.dtype.name, None)


def _get_cv2_norm_name(norm: int) -> str:
    """
    Get the string representation of an OpenCV normalization type constant.
    
    Args:
        norm (int): OpenCV normalization type constant
        
    Returns:
        str: String representation of the normalization type or None if not found
    """
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
    """
    Convert OpenCV dtype to NumPy dtype string representation.
    
    Args:
        dtype (int): OpenCV dtype constant
        
    Returns:
        str: String representation of the corresponding NumPy dtype or None if not found
    """
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
    """
    Log array shape, dtype, and min/max values for debugging purposes.
    
    Args:
        array (np.ndarray): Input array to analyze
        
    Returns:
        np.ndarray: The input array unchanged
    """
    logger.debug(f"{array.shape=} {array.dtype=}")
    logger.debug(f"min = {float(array.min())}; max = {float(array.max())}")
    return array


def _tofromimage(func, **kwargs):
    """
    Decorator to handle conversion between Image objects and arrays.
    
    Extracts affine transformation and coordinate reference system (CRS) from Image object,
    applies the function to the image data, and reconstructs an Image object with the result.
    
    Args:
        func: Function to be wrapped
        **kwargs: Additional keyword arguments
        
    Returns:
        callable: Wrapped function that handles Image object conversion
    """
    def inner(**kwargs):
        affine = kwargs["array"].affine
        crs = kwargs["array"].crs
        kwargs["array"] = kwargs["array"].image

        altered_array = func(**kwargs)
        return Image(altered_array, affine, crs)

    return inner


def _is_cv8u(array: np.ndarray) -> bool:
    """
    Check if array has 8-bit unsigned integer type.
    
    Args:
        array (np.ndarray): Input array to check
        
    Returns:
        bool: True if array is 8-bit unsigned integer type, False otherwise
    """
    return _np_to_cv2_dtype(array) == cv2.CV_8U


def _is_multiband(array) -> bool:
    """
    Check if array has more than 2 dimensions (is multi-band).
    
    Args:
        array: Input array to check
        
    Returns:
        bool: True if array has more than 2 dimensions, False otherwise
    """
    return array.ndim > 2


def _normalize(array, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U):
    """
    Normalize array values to a specified range using OpenCV normalization.
    
    Args:
        array: Input array to normalize
        alpha (float): Lower bound of normalization range
        beta (float): Upper bound of normalization range
        norm_type: OpenCV normalization type constant
        dtype: Output data type
        
    Returns:
        np.ndarray: Normalized array
        
    Raises:
        TypeError: If input is not a NumPy array
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    logger.debug(
        f"Normalizing image to range: ({alpha}, {beta})." +
        f" Normalization: {_get_cv2_norm_name(cv2.NORM_MINMAX)}." +
        f" Output format: {_cv2_to_np_dtype(dtype)}")

    return cv2.normalize(array, dst=None, alpha=alpha, beta=beta,
                         norm_type=norm_type, dtype=dtype)


def _overwrite_nodata(array, nodata: int | float = -9999.) -> np.ndarray:
    """
    Replace specified nodata values with DEFAULT_NODATA.
    
    Args:
        array (np.ndarray): Input array
        nodata (int | float): Value to be treated as nodata
        
    Returns:
        np.ndarray: Array with nodata values replaced
    """
    logger.debug(
        f"Values equal to {nodata} treated as NULL (overwritten as DEFAULT_NODATA)")
    array = array.copy()
    array[array == nodata] = DEFAULT_NODATA
    return array


def to_single_band_uint8(array, **kwargs) -> np.ndarray:
    """
    Convert array to single-band 8-bit unsigned integer format.
    
    If input is multi-band, converts to grayscale. Normalizes values to 0-255 range.
    
    Args:
        array (np.ndarray): Input array
        **kwargs: Additional keyword arguments
        
    Returns:
        np.ndarray: 8-bit unsigned integer single-band array
    """
    _array_verbose(array)

    if not _is_cv8u(array):
        logger.debug("Converting dtype to CV_8U")
        array = _normalize(array)

    if _is_multiband(array):
        logger.debug("Multi-band array (ndim > 2)")
        array = _normalize(color.rgb2gray(array))

    return array


def svd_filter(patch: np.ndarray, k: int = 5, full_matrices=False):
    """
    Apply Singular Value Decomposition (SVD) filter to an image patch.
    
    Performs dimensionality reduction by keeping only the k largest singular values.
    
    Args:
        patch (np.ndarray): Input image patch
        k (int): Number of singular values to keep
        full_matrices (bool): Whether to compute full matrices in SVD
        
    Returns:
        np.ndarray: Filtered image patch
    """
    U, stride, VT = np.linalg.svd(patch, full_matrices=full_matrices)

    # Keep only the first k singular values
    S_filtered = np.zeros_like(stride)
    S_filtered[:k] = stride[:k]
    S_matrix = np.diag(S_filtered)

    # Reconstruct the patch
    return np.dot(U, np.dot(S_matrix, VT))