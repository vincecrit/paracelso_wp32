import logging

import numpy as np
from skimage import color, exposure

from ot.image_processing import common

logger = logging.getLogger(__name__)


__all__ = [
    "equalize",
    "minmax",
    "zscore",
    "lognorm",
    "clahe"
]


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


def _prevent_multiband(array):
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
    return _prevent_multiband(equalized)


@common._tofromimage
def clahe(*, array: np.ndarray,
          kernel_size: float | tuple = 3,
          clip_limit: float = 0.05) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.
    Parameters
    ----------
    array : np.ndarray
        Input image array.
    kernel_size : float or tuple, optional
        Size of the kernel used for the local histogram equalization. 
        If a single float is provided, it is used for both dimensions. 
        Default is 10.
    clip_limit : float, optional
        Clipping limit for contrast enhancement. Higher values give more contrast.
        Default is 0.01.
    Returns
    -------
    np.ndarray
        Image array after applying CLAHE.
    Notes
    -----
    This function uses `skimage.exposure.equalize_adapthist` to perform the 
    adaptive histogram equalization. The input image array is first rescaled 
    to float intensity before applying CLAHE. The resulting image is then 
    processed to prevent multiband issues.
    """
    logger.info("Eseguo skimage.exposure.equalize_adapthist")
    logger.debug(f"CLAHE finestra mobile: {kernel_size=}")
    common._array_verbose(array)

    array = _rescale_float_intensity(array)
    clahe_array = exposure.equalize_adapthist(array, kernel_size, clip_limit)
    return _prevent_multiband(clahe_array)


@common._tofromimage
def lognorm(*, array: np.ndarray, gain: float = 1.) -> np.ndarray:
    logger.info("Eseguo skimage.exposure.adjust_log")
    common._array_verbose(array)

    array = _rescale_float_intensity(array, in_range=(0, 1))
    adjusted_array = exposure.adjust_log(array, gain=gain)
    return _prevent_multiband(adjusted_array)


@common._tofromimage
def zscore(*, array: np.ndarray) -> np.ndarray:
    """
    Normalize a band using z-score normalization.
    """
    logger.info("Eseguo trasformata `zscore` con metodi skimage")
    common._array_verbose(array)

    # array = _rescale_float_intensity(array, in_range=(-1, 1))
    mean, std = np.mean(array), np.std(array)
    normalized = (array - mean) / std
    return _prevent_multiband(normalized)


@common._tofromimage
def minmax(*, array: np.ndarray) -> np.ndarray:
    """
    Normalize a band by its minimum and maximum values, maintaining NoData values.
    """
    logger.info("Eseguo normalizzazione rispetto ai valori minimo e massimo")
    common._array_verbose(array)

    normalized = _rescale_float_intensity(array, in_range=(array.min(), array.max()))
    return _prevent_multiband(normalized)