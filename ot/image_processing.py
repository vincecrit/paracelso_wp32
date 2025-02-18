"""
This module provides various functions for normalizing and transforming 2D arrays and image data. 
It includes utilities for applying rolling windows, converting arrays to 8-bit unsigned integers, 
equalizing histograms of image channels, and performing different types of normalization such as 
standard normalization, power transformation, and z-score normalization. Additionally, it offers 
functions to handle raster data using the rasterio library.
Functions:
- stepped_rolling_window(array_2d: np.ndarray, win_size: tuple, step_size: tuple = (1, 1)) -> tuple[np.ndarray]:
    Apply a rolling window with step size on a 2D array.
- _to_CV8U(a: np.ndarray, cv2_norm_type: int = cv2.NORM_MINMAX) -> np.ndarray:
    Convert an array to 8-bit unsigned integer using OpenCV normalization.
- cv2_equalize_channels(array: np.ndarray) -> np.ndarray:
    Equalize the histogram of each channel in an array using OpenCV.
- rasterio_to_CV2_8U(source: str):
    Convert a rasterio image to an 8-bit unsigned integer image using OpenCV.
- powernorm(arr: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    Normalize an array by applying a power transformation.
- _normalize_band(band, mask=None, nodata: int | float | None = np.nan):
    Normalize a band by its minimum and maximum values, maintaining NoData values.
- _zscore_band(band, mask=None, nodata: int | float | None = np.nan):
    Normalize a band using z-score normalization, maintaining NoData values.
- _log_band(band, mask=None, epsilon=1e-5, nodata: int | float | None = np.nan):
    Normalize a band using logarithmic transformation, maintaining NoData values.
- _clahe(band, clip_limit: float = 2., kernel_size: int | tuple[int] = 3, mask=None, nodata: int | float | None = np.nan):
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to a band, maintaining NoData values.
"""
from itertools import product

import cv2
import numpy as np
import rasterio
from numpy.lib.stride_tricks import as_strided


def stepped_rolling_window(array_2d: np.ndarray, win_size: tuple,
                           step_size: tuple = (1, 1)) -> tuple[np.ndarray]:
    """Apply a rolling window with step size on a 2D array.

    Args:
        array_2d (np.ndarray): Array 2D di input.
        win_size (tuple): Dimensione della finestra (win_h, win_w).
        step_size (tuple): Passo della finestra (step_h, step_w).
        Di default è pari a `(1, 1)`.

    Returns:
        tuple[np.ndarray]: (1) indici del centro di ogni finestra mobile.
        (2) Array 3D con viste delle finestre (n_rows*n_cols, win_h, win_w).
    """
    win_h, win_w = win_size
    step_h, step_w = step_size
    arr_h, arr_w = array_2d.shape

    # Numero di finestre lungo ogni dimensione
    out_shape = ((arr_h - win_h) // step_h + 1,
                 (arr_w - win_w) // step_w + 1, win_h, win_w)

    # Strided views
    strides = (array_2d.strides[0] * step_h, array_2d.strides[1]
               * step_w, array_2d.strides[0], array_2d.strides[1])

    i = np.arange(step_size[0], array_2d.shape[0], step_size[0]).astype(int)
    j = np.arange(step_size[1], array_2d.shape[1], step_size[1]).astype(int)

    indexes = np.array(list(product(i, j)))

    return indexes, as_strided(array_2d, shape=out_shape,
                               strides=strides).reshape(-1, *win_size)


def _to_CV8U(a: np.ndarray, cv2_norm_type: int = cv2.NORM_MINMAX) -> np.ndarray:
    """Convert an array to 8-bit unsigned integer using OpenCV normalization."""
    return cv2.normalize(a, alpha=0, beta=255, dst=None, norm_type=cv2_norm_type, dtype=cv2.CV_8U)


def cv2_equalize_channels(array: np.ndarray) -> np.ndarray:
    """Equalize the histogram of each channel in an array using OpenCV."""
    try:
        *(rows, cols), channels = array.shape
        print(rows, cols, channels)

    except ValueError:
        print('non ha 3 canali')
        return cv2_equalize_channels(array.reshape(*array.shape, 1))

    eq = list()
    for channel in range(channels):
        eq.append(cv2.equalizeHist(array[..., channel]))

    return cv2.merge(eq)


def rasterio_to_CV2_8U(source: str):
    """Convert a rasterio image to an 8-bit unsigned integer image using OpenCV."""
    with rasterio.open(source) as src:
        print(src.meta)
        bands = []
        for b in range(src.count):
            bands.append(_to_CV8U(src.read(b + 1)))

        return cv2.merge(bands)


def powernorm(band: np.ndarray, gamma: float = 1.0, mask=None, nodata: int | float | None = np.nan) -> np.ndarray:
    """Normalize an array by applying a power transformation.

    Parameters:
    arr (array-like): L'array di input da normalizzare.
    gamma (float): Il parametro della trasformazione di potenza.

    Returns:
    np.ndarray: L'array normalizzato.
    """
    if mask is None:
        mask = np.zeros_like(band).astype(bool)
    min_val, max_val = band.min(), band.max()

    if max_val == min_val:
        return np.zeros_like(band)

    normalized = (band - min_val) / (max_val - min_val)
    normalized = normalized ** gamma
    normalized[mask] = nodata  # Mantieni NoData
    return normalized


def _normalize_band(band, mask=None, nodata: int | float | None = np.nan):
    """Normalize a band by its minimum and maximum values, maintaining NoData values."""
    if mask is None:
        mask = np.zeros_like(band).astype(bool)
    valid_pixels = band[~mask]
    min_val, max_val = np.min(valid_pixels), np.max(valid_pixels)
    normalized = (band - min_val) / (max_val - min_val)
    normalized[mask] = nodata  # Mantieni NoData
    return normalized


def _zscore_band(band, mask=None, nodata: int | float | None = np.nan):
    """Normalize a band using z-score normalization, maintaining NoData values."""
    if mask is None:
        mask = np.zeros_like(band).astype(bool)
    valid_pixels = band[~mask]
    mean, std = np.mean(valid_pixels), np.std(valid_pixels)
    normalized = (band - mean) / std
    normalized[mask] = nodata
    return normalized


def _log_band(band, mask=None, epsilon=1e-5, nodata: int | float | None = np.nan):
    """Normalize a band using logarithmic transformation, maintaining NoData values."""
    if mask is None:
        mask = np.zeros_like(band).astype(bool)
    normalized = np.log(band + epsilon)
    normalized[mask] = nodata
    return normalized


def _clahe(band, clip_limit: float = 2., kernel_size: int | tuple[int] = 3,
           mask=None, nodata: int | float | None = np.nan):
    """Contrast limited adaptive histogram equalization"""
    if mask is None:
        mask = np.zeros_like(band).astype(bool)
    
    if isinstance(kernel_size, int):
        kernel_size = kernel_size, kernel_size
        
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=kernel_size)
    clahe_band = clahe.apply(band)
    clahe_band[mask] = nodata
    return clahe_band