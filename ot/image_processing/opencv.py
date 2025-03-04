import logging

import cv2
import numpy as np

from ot.image_processing import common

logger = logging.getLogger(__name__)


@common._tofromimage
def clahe(*, array: np.ndarray, clip_limit: float = 2.,
          kernel_size: int | tuple[int] = 3) -> np.ndarray:
    """
    Contrast limited adaptive histogram equalization
    """
    logger.info("Eseguo CLAHE con metodi OpenCV")
    common._array_verbose(array)
    
    array = common.np2cv(array)

    if isinstance(kernel_size, int):
        kernel_size = kernel_size, kernel_size
        logger.debug(f"Finestra mobile: {kernel_size=}")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=kernel_size)
    return clahe.apply(array)


@common._tofromimage
def equalize(*, array):
    """
    Basic histogram equalization.
    """
    logger.info("Eseguo cv2.equalizeHist")
    array = common.np2cv(array=array)
    array = cv2.equalizeHist(array)
    return array


@common._tofromimage
def lognorm(*, array: np.ndarray, gain: float = 1.):
    """
    Normalize a band using logarithmic transformation.
    """
    logger.info("Eseguo trasformazione logaritmica con metodi OpenCV")

    array_dtype = common._np_to_cv2_dtype(array)
    array = common.np2cv(array, nodata=-9999.)

    logger.debug(f"Converto il formato in ingresso in {cv2.CV_32F=} prima della trasformazione logaritmica")
    array = common._normalize(array, 0, 1, dtype=cv2.CV_32F)

    normalized = gain * np.log(1. + array)

    return common._toCV8U(normalized)
