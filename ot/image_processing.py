"""
This module provides various functions for normalizing and transforming 2D arrays and image data. 
It includes utilities for applying rolling windows, converting arrays to 8-bit unsigned integers, 
equalizing histograms of image channels, and performing different types of normalization such as 
standard normalization, power transformation, and z-score normalization. Additionally, it offers 
functions to handle raster data using the rasterio library.
Functions:
- stepped_rolling_window(array_2d: np.ndarray, win_size: tuple, step_size: int) -> tuple[np.ndarray]:
    Apply a rolling window with step size on a 2D array.
- toCV8U(a: np.ndarray, cv2_norm_type: int = cv2.NORM_MINMAX) -> np.ndarray:
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
import cv2
import numpy as np
import rasterio

from skimage.util import view_as_windows


def stepped_rolling_window(array_2d: np.ndarray, window_shape: tuple[int], step: int):
    """
    Restituisce le finestre mobili dell'array di input utilizzando la funzione
    `skimage.utils.view_as_windows` e gli indici corrispondenti al centro di ogni
    finestra.

    Args:
        array_2d (np.ndarray): Array 2D di input.
        window_shape (tuple): Dimensione della finestra (altezza, larghezza).
        step (int): Passo di campionamento, considerato omogeneo nelle 2 direzioni.

    Returns:
        windows (np.ndarray): finestre mobili (NxM, *window_shape)
        centers (np.ndarray): indici corrispondenti ai centri delle finestre (NxM, 2).
    """
    windows = view_as_windows(
        arr_in=array_2d, window_shape=window_shape, step=step)

    # Calcolo dell'offset per il centro della finestra
    center_offset_row = window_shape[0] // 2
    center_offset_col = window_shape[1] // 2

    centers = []
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            center_row = i * step + center_offset_row
            center_col = j * step + center_offset_col
            centers.append((center_row, center_col))

    return windows.reshape(-1, *window_shape), np.array(centers)


def toCV8U(a: np.ndarray, cv2_norm_type: int = cv2.NORM_MINMAX) -> np.ndarray:
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


def rasterio2cv2(source: str, band: int | None = None) -> np.ndarray:
    """Convert a rasterio image to an 8-bit unsigned integer image using OpenCV."""

    with rasterio.open(source) as src:
        if band is None:
            iter_bands = range(src.count)
        else:
            iter_bands = [band]

        channels = []
        for b in iter_bands:
            band = src.read(b+1)
            band[band == src.meta['nodata']] = 0
            channels.append(band)

        return cv2.cvtColor(cv2.merge(channels), cv2.COLOR_RGB2BGR)


def power_norm(band: np.ndarray, gamma: float = 1.0, mask=None,
               nodata: int | float | None = np.nan) -> np.ndarray:
    """Normalize an array by applying a power transformation.

    Parameters:
    arr (array-like): L'array di input da normalizzare.
    gamma (float): Il parametro della trasformazione di potenza.

    Returns:
    np.ndarray: L'array normalizzato.
    """
    if mask is None:
        mask = np.zeros_like(band).astype(bool)

    min_val, max_val = band[~mask].min(), band[~mask].max()

    if max_val == min_val:
        return np.zeros_like(band)

    normalized = (band - min_val) / (max_val - min_val)
    normalized = normalized ** gamma
    normalized[mask] = nodata  # Mantieni NoData

    return normalized


def norm_minmax(band, mask=None, nodata: int | float | None = np.nan):
    """Normalize a band by its minimum and maximum values, maintaining NoData values."""

    if mask is None:
        mask = np.zeros_like(band).astype(bool)

    valid_pixels = band[~mask]
    min_val, max_val = np.min(valid_pixels), np.max(valid_pixels)
    normalized = (band - min_val) / (max_val - min_val)
    normalized[mask] = nodata  # Mantieni NoData

    return normalized


def norm_zscore(band, mask=None, nodata: int | float | None = np.nan):
    """Normalize a band using z-score normalization, maintaining NoData values."""
    if mask is None:
        mask = np.zeros_like(band).astype(bool)

    valid_pixels = band[~mask]
    mean, std = np.mean(valid_pixels), np.std(valid_pixels)
    normalized = (band - mean) / std
    normalized[mask] = nodata

    return normalized


def norm_log(band: np.ndarray, mask=None, epsilon=1e-5, nodata: int | float | None = np.nan):
    """Normalize a band using logarithmic transformation, maintaining NoData values."""

    if mask is None:
        mask = np.zeros_like(band).astype(bool)

    normalized = np.log(band + epsilon)
    normalized[mask] = nodata
    return normalized


def cv2_clahe(band: np.ndarray, clip_limit: float = 2., kernel_size: int | tuple[int] = 3,
              mask=None, nodata: int | float | None = 0):
    """Contrast limited adaptive histogram equalization"""

    band = cv2.normalize(band, dst=None, alpha=0, beta=255,
                         norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    if mask is None:
        mask = np.zeros_like(band).astype(bool)

    if isinstance(kernel_size, int):
        kernel_size = kernel_size, kernel_size

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=kernel_size)
    clahe_band = clahe.apply(band)
    clahe_band[mask] = nodata
    return clahe_band
