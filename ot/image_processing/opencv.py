import logging

import cv2
import numpy as np

from ot.image_processing import common

from log import setup_logger

logger = setup_logger()

__all__ = [
    "equalize",
    "minmax",
    "zscore",
    "lognorm",
    "clahe"
]


@common._tofromimage
def clahe(*, array: np.ndarray,
          clip_limit: float = .05,
          kernel_size: int | tuple[int] = 3) -> np.ndarray:
    """
    Contrast limited adaptive histogram equalization
    """
    logger.info("Eseguo CLAHE con metodi OpenCV")
    common._array_verbose(array)

    array = common.to_single_band_uint8(array)

    if isinstance(kernel_size, int):
        kernel_size = kernel_size, kernel_size

    logger.debug(f"CLAHE finestra mobile: {kernel_size=}")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=kernel_size)
    return clahe.apply(array)


@common._tofromimage
def equalize(*, array):
    """
    Basic histogram equalization.
    """
    logger.info("Eseguo cv2.equalizeHist")
    array = common.to_single_band_uint8(array=array)
    array = cv2.equalizeHist(array)
    return array


@common._tofromimage
def lognorm(*, array: np.ndarray,
            gain: float = 1.):
    """
    Normalize a band using logarithmic transformation.
    """
    logger.info("Eseguo trasformazione logaritmica con metodi OpenCV")
    common._array_verbose(array)

    array = common.to_single_band_uint8(array, nodata=-9999.)

    logger.debug(
        f"Converto il formato in ingresso in `CV_32F` prima della trasformazione logaritmica")
    array = common._normalize(array, 0, 1, dtype=cv2.CV_32F)

    normalized = gain * np.log(1. + array)

    return common._normalize(normalized, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


@common._tofromimage
def zscore(*, array):
    logger.info("Applico la trasformata zscore con metodi OpenCV")
    array = common.to_single_band_uint8(array)
    common._array_verbose(array)

    mean, std = np.mean(array), np.std(array)
    normalized = (array - mean) / std

    return common._normalize(normalized)


@common._tofromimage
def minmax(*, array):
    logger.info("Scalo le intensità sui valori minimo/massimo con metodi OpenCV")
    common._array_verbose(array)

    return common.to_single_band_uint8(array)
