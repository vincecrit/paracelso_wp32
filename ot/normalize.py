import numpy as np

__all__ = [
    'normalize', 'powernorm', 'lognorm', 'rgb2single_band'
]


def std_norm(arraylike: np.ndarray) -> np.ndarray:
    """
    Normalizza i valori dell'array sottraendo la media e dividendo per la deviazione standard.
    """
    return (arraylike - np.mean(arraylike))/np.std(arraylike)


def normalize(arraylike: np.ndarray) -> np.ndarray:
    """
    Normalizza i valori dell'array in un range tra 0 e 1.
    
    Argomenti:
    arraylike (np.ndarray): input array da normalizzare.

    Returns:
    np.ndarray: array normalizzato con valori nell'intervallo [0, 1].
    """

    return (arraylike - np.min(arraylike))/(np.max(arraylike) - np.min(arraylike))


def to_uint8(floatarray: np.ndarray) -> np.ndarray:
    """
    Converte un array di float in un array di interi unsigned 8-bit.
    """
    return (floatarray*255).astype('uint8')


def powernorm(arr: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Normalizza un array applicando una trasformazione di potenza.

    Parameters:
    arr (array-like): L'array di input da normalizzare.
    gamma (float): Il parametro della trasformazione di potenza.

    Returns:
    np.ndarray: L'array normalizzato.
    """
    # arr = np.asarray(arr, dtype=np.float32)
    min_val, max_val = arr.min(), arr.max()

    if max_val == min_val:
        return np.zeros_like(arr)

    normalized = (arr - min_val) / (max_val - min_val)
    return normalized ** gamma


def lognorm(arr):
    """
    Normalizza un array applicando una trasformazione logaritmica.

    Parameters:
    arr (array-like): L'array di input da normalizzare.

    Returns:
    np.ndarray: L'array normalizzato.
    """
    # arr = np.asarray(arr, dtype=np.float64)
    min_val = arr.min()

    if min_val <= 0:
        arr = arr - min_val + 1  # Shift per evitare log(0) o valori negativi

    return np.log1p(arr - arr.min()) / np.log1p(arr.max() - arr.min())


def rgb2single_band(image: np.ndarray):
    '''
    Converte un'immagine RGB in scala di grigi utilizzando la formula:

    Y709 = 0.2125*R + 0.7154*G + 0.0721*B
    '''
    if not image.ndim == 3:
        raise ValueError("Input image must be a 3D array")

    R, G, B = 0.21250, 0.71540, 0.07210
    b1, b2, b3 = image

    b1 *= R
    b2 *= G
    b3 *= B

    return b1 + b2 + b3
