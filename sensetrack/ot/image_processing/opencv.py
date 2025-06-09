"""
This module provides OpenCV-based image processing operations including histogram equalization,
normalization, and denoising methods.
"""
import cv2
import numpy as np

from sensetrack.log import setup_logger
from sensetrack.ot.image_processing import common

logger = setup_logger(__name__)

__all__ = [
    "equalize",
    "minmax",
    "zscore",
    "lognorm",
    "clahe",
    "svd_denoise"
]


@common._tofromimage
def clahe(*, array: np.ndarray,
          clip_limit: float = .05,
          kernel_size: int | tuple[int] = 3) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

    Args:
        array (np.ndarray): Input image array
        clip_limit (float): Threshold for contrast limiting. Default is 0.05
        kernel_size (int | tuple[int]): Size of grid for histogram equalization.
            If int, same value is used for both dimensions.
            If tuple, specifies (width, height). Default is 3

    Returns:
        np.ndarray: CLAHE enhanced image
    """
    logger.info("Applying CLAHE using OpenCV methods")
    array = common.to_single_band_uint8(array)

    if isinstance(kernel_size, int):
        kernel_size = kernel_size, kernel_size

    logger.debug(f"CLAHE sliding window: {kernel_size=}")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=kernel_size)
    return clahe.apply(array)


@common._tofromimage
def equalize(*, array):
    """
    Apply basic histogram equalization to an image.

    This function enhances the contrast of the image by effectively
    spreading out the most frequent intensity values.

    Args:
        array: Input image array

    Returns:
        np.ndarray: Histogram equalized image
    """
    logger.info("Applying cv2.equalizeHist")
    array = common.to_single_band_uint8(array=array)
    array = cv2.equalizeHist(array)
    return array


@common._tofromimage
def lognorm(*, array: np.ndarray,
            gain: float = 1.):
    """
    Apply logarithmic normalization to an image.

    This transformation enhances details in darker regions while compressing
    the dynamic range of brighter regions.

    Args:
        array (np.ndarray): Input image array
        gain (float): Multiplication factor for the log transform. Default is 1.0

    Returns:
        np.ndarray: Log-normalized image
    """
    logger.info("Applying logarithmic transformation using OpenCV methods")
    common._array_verbose(array)

    array = common.to_single_band_uint8(array, nodata=-9999.)

    logger.debug(
        f"Converting input format to `CV_32F` before logarithmic transformation")
    array = common._normalize(array, 0, 1, dtype=cv2.CV_32F)

    normalized = gain * np.log(1. + array)

    return common._normalize(normalized, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


@common._tofromimage
def zscore(*, array):
    """
    Apply Z-score normalization to an image.

    Normalizes the image by subtracting the mean and dividing by the standard deviation,
    resulting in a distribution with zero mean and unit variance.

    Args:
        array: Input image array

    Returns:
        np.ndarray: Z-score normalized image
    """
    logger.info("Applying zscore transform using OpenCV methods")
    array = common.to_single_band_uint8(array)
    common._array_verbose(array)

    mean, std = np.mean(array), np.std(array)
    normalized = (array - mean) / std

    return common._normalize(normalized)


@common._tofromimage
def minmax(*, array):
    """
    Apply min-max scaling to an image.

    Scales the image values to the full range of the 8-bit unsigned integer format (0-255).

    Args:
        array: Input image array

    Returns:
        np.ndarray: Min-max scaled image
    """
    logger.info("Scaling intensities to min/max values using OpenCV methods")
    common._array_verbose(array)

    return common.to_single_band_uint8(array)


@common._tofromimage
def svd_denoise(array: np.ndarray, patch_size: int | None = None, stride: int| None = None, k=3) -> np.ndarray:
    """
    Apply Singular Value Decomposition (SVD) based denoising using a sliding window.

    This method performs denoising by applying SVD to local patches of the image
    and reconstructing them using only the k most significant singular values.

    Args:
        array (np.ndarray): Input image array
        patch_size (int, optional): Size of the sliding window. If None, automatically calculated
            based on image dimensions
        stride (int, optional): Distance between windows (overlap = patch_size - stride).
            If None, set to half the patch size
        k (int): Number of singular values to keep for reconstruction. Default is 3

    Returns:
        np.ndarray: Denoised image array
    """
    array = common.to_single_band_uint8(array, nodata=-9999.)
    
    logger.info("Applying SVD denoising")
    if patch_size is None:
        raxis = min(array.shape) # reference axis
        n = float(10**(np.floor(np.log10(raxis))-1))
        patch_size = int(round((raxis // 10) / n, 0) * n)
    
    if stride is None:
        stride = int(round(patch_size / 2, 0))
    
    logger.info(f"{patch_size=} {stride=}")
    # Padding
    pad_r = (patch_size - array.shape[0] % stride) % stride
    pad_c = (patch_size - array.shape[1] % stride) % stride
    
    padded_array = np.pad(
        common._normalize(array, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F),
        ((0, pad_r), (0, pad_c)),
        mode='reflect')
    
    # Empty copy for reconstructed image
    reconstructed = np.zeros_like(padded_array)
    count_matrix = np.zeros_like(padded_array)

    # Loop over windows
    for i in range(0, padded_array.shape[0] - patch_size + 1, stride):
        for j in range(0, padded_array.shape[1] - patch_size + 1, stride):

            patch = padded_array[i:i+patch_size, j:j+patch_size]
            patch_reconstructed = common.svd_filter(patch, k)

            # Insert reconstructed patch into new image (average in overlapping points)
            reconstructed[i:i+patch_size, j:j+patch_size] += patch_reconstructed
            count_matrix[i:i+patch_size, j:j+patch_size] += 1

    # Normalize pixels in overlapping areas
    reconstructed =  (reconstructed / count_matrix)[:array.shape[0], :array.shape[1]]
    return common._normalize(reconstructed, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)