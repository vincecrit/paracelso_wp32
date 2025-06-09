import numpy as np
from skimage import color, exposure

from sensetrack.log import setup_logger
from sensetrack.ot.image_processing import common

logger = setup_logger(__name__)


__all__ = [
    "equalize",
    "minmax",
    "zscore",
    "lognorm",
    "clahe"
]


def _rescale_float_intensity(array: np.ndarray, in_range=(-1, 1)) -> np.ndarray:
    """
    Rescales the intensity of a NumPy array with a floating-point data type to a specified range.
    This function creates a copy of the input array and, if its data type is a floating-point type
    (float8, float16, float32, or float64), rescales its intensity values to the specified `in_range`
    using `skimage.exposure.rescale_intensity`. The function logs the operation and provides verbose
    output about the resulting array.

    Arguments:
        array (numpy.ndarray): Input NumPy array whose intensity values are to be rescaled.
        in_range (tuple, optional): The intensity range to which the input array will be rescaled.
        Default is (-1, 1).

    Returns:
        np.ndarray: The rescaled NumPy array.
    """
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
    """
    Converts a multiband (e.g., RGB) image array to a single-band grayscale image if necessary.
    Arguments:
        array (numpy.ndarray): Input image array, which may be single-band (grayscale) or multiband (e.g., RGB).
    Returns:
        numpy.ndarray: A single-band (grayscale) image array. If the input was already single-band, returns a copy of the original array.
    Notes:
        - Uses `common._is_multiband` to check if the input array is multiband.
        - If the input is multiband, converts it to grayscale using `color.rgb2gray`.
        - Logs a debug message if conversion is performed.
    """
    array = array.copy()

    if common._is_multiband(array):
        logger.debug("Immagine multibanda. Eseguo RGB2GRAY")
        array = color.rgb2gray(array)
        
    return array


@common._tofromimage
def equalize(*, array: np.ndarray) -> np.ndarray:
    """
    Applies histogram equalization to the input image array.
    This function uses `skimage.exposure.equalize_hist` to enhance the
    contrast of the image by redistributing pixel intensities. 
    It logs the operation and provides verbose information about the input array.
    The result is processed to prevent multiband output if necessary.
    Arguments:
        array (np.ndarray): The input image array to be equalized.
    Returns:
        np.ndarray: The histogram-equalized image array.
    """
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
    """
    Applies logarithmic normalization to an input image array.
    This function rescales the input array to the [0, 1] range and then applies
    logarithmic intensity adjustment using the specified gain. It is typically used
    to enhance the visibility of features in images with a large dynamic range.
    Arguments:
        array (np.ndarray): Input image array to be normalized.
        gain (float, optional): Multiplicative gain factor for the logarithmic
            adjustment. Default is 1.0.
    Returns:
        np.ndarray: The log-normalized image array with the same shape as the input.
    Notes:
        - The input array is first rescaled to the [0, 1] range before applying
        the logarithmic adjustment.
        - The function ensures that the output maintains the correct number of bands
        for multiband images.
    """
    logger.info("Eseguo skimage.exposure.adjust_log")
    common._array_verbose(array)

    array = _rescale_float_intensity(array, in_range=(0, 1))
    adjusted_array = exposure.adjust_log(array, gain=gain)
    return _prevent_multiband(adjusted_array)


@common._tofromimage
def zscore(*, array: np.ndarray) -> np.ndarray:
    """
    Applies z-score normalization to the input array.
    This function normalizes the input NumPy array by subtracting its mean and dividing by its standard deviation,
    resulting in an array with zero mean and unit variance. It is typically used to standardize image bands or
    other numerical data for further processing.
    Arguments:
        array (np.ndarray): The input array to be normalized.
    Returns:
        np.ndarray: The z-score normalized array.
    Raises:
        ValueError: If the input array has zero standard deviation.
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
    Normalizes the input array to the range [0, 1] based on its minimum and maximum values,
    while preserving NoData values.
    Arguments:
        array (np.ndarray): The input array (band) to be normalized.
    Returns:
        np.ndarray: The normalized array with values scaled between 0 and 1.
    """
    logger.info("Eseguo normalizzazione rispetto ai valori minimo e massimo")
    common._array_verbose(array)

    normalized = _rescale_float_intensity(array, in_range=(array.min(), array.max()))
    return _prevent_multiband(normalized)