import logging

import numpy as np
from skimage import exposure
from skimage import color

from ot.image_processing import common

logger = logging.getLogger(__name__)


def _rescale_float_intensity(array: np.ndarray, in_range=(-1, 1)) -> np.ndarray:
    array = array.copy()
    accepted_dtypes = [f"float{i}" for i in [8, 16, 32, 64]]

    if array.dtype in accepted_dtypes:
        msg = f"Conversione array con"+\
            f"skimage.exposure.rescale_intensity {in_range = }"
        
        logger.info(msg)

        array = exposure.rescale_intensity(array, in_range=in_range)
    else:
        pass

    common._array_verbose(array)

    return array


def prevent_multiband(array):
    array = array.copy()

    if common._is_multiband(array):
        logger.debug("Immagine multibanda. Eseguo RGB2GRAY")
        array = color.rgb2gray(array)
        
    return array


@common._tofromimage
def equalize(*, array: np.ndarray) -> np.ndarray:
    logger.info("Eseguo skimage.exposure.equalize_hist")
    common._array_verbose(array)

    equalized = exposure.equalize_hist(array)
    return prevent_multiband(equalized)


@common._tofromimage
def clahe(*, array: np.ndarray,
          kernel_size: float | tuple = 10,
          clip_limit: float = 0.01) -> np.ndarray:
    logger.info("Eseguo skimage.exposure.equalize_adapthist")
    common._array_verbose(array)

    array = _rescale_float_intensity(array)
    clahe_array = exposure.equalize_adapthist(array, kernel_size, clip_limit)
    return prevent_multiband(clahe_array)


@common._tofromimage
def lognorm(*, array: np.ndarray, gain: float = 1.) -> np.ndarray:
    logger.info("Eseguo skimage.exposure.adjust_log")
    common._array_verbose(array)

    array = _rescale_float_intensity(array, in_range=(0, 1))
    adjusted_array = exposure.adjust_log(array, gain=gain)
    return prevent_multiband(adjusted_array)
