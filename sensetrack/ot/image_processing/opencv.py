import cv2
import numpy as np

from log import setup_logger
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
    Contrast limited adaptive histogram equalization
    """
    logger.info("Eseguo CLAHE con metodi OpenCV")
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


@common._tofromimage
def svd_denoise(array: np.ndarray, patch_size: int | None = None, stride: int| None = None, k=3) -> np.ndarray:
    """
    Applica la Singolar Value Decomposition su finestra mobile ad un'immagine.
    
    Args:
        array(numpy.ndarray): array
        patch_size(int): passo della finestra mobile
        stride(int): distanza tra le finestre (sovrapposizione = patch_size - stride)
        k(int): numero di valori di singoli da considerare (default = 10)

    Returns:
        reconstructed(numpy.ndarray): array ricostruito con k valori singoli
    """
    array = common.to_single_band_uint8(array, nodata=-9999.)
    
    logger.info("Applico denoise con SVD")
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
    
    # Copia vuota per l'immagine ricostruita
    reconstructed = np.zeros_like(padded_array)
    count_matrix = np.zeros_like(padded_array)

    # Loop sulle finestre
    for i in range(0, padded_array.shape[0] - patch_size + 1, stride):
        for j in range(0, padded_array.shape[1] - patch_size + 1, stride):

            patch = padded_array[i:i+patch_size, j:j+patch_size]
            patch_reconstructed = common.svd_filter(patch, k)

            # Inserisci la patch ricostruita nella nuova immagine (media nei punti sovrapposti)
            reconstructed[i:i+patch_size, j:j+patch_size] += patch_reconstructed
            count_matrix[i:i+patch_size, j:j+patch_size] += 1

    # Normalizza i pixel nelle aree sovrapposte
    reconstructed =  (reconstructed / count_matrix)[:array.shape[0], :array.shape[1]]
    return common._normalize(reconstructed, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)