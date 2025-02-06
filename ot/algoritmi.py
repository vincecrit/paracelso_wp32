import json
import sys
from enum import Enum, unique
from itertools import product
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from geopandas import points_from_xy
from numpy.lib.stride_tricks import as_strided
from rasterio.transform import AffineTransformer
from skimage.registration import (optical_flow_ilk, optical_flow_tvl1,
                                  phase_cross_correlation)

from ot.interfaces import OTAlgorithm


def to_uint8(floatarray: np.ndarray) -> np.ndarray:
    return (floatarray*255).astype('uint8')


def cv2_equalize_channels(array: np.ndarray) -> np.ndarray:
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


def rasterio_to_cv2(source: str):
    with rasterio.open(source) as src:
        print(src.meta)
        bands = []
        for b in range(src.count):
            bands.append(to_uint8(src.read(b + 1)))

        return cv2.merge(bands)


class Image:
    def __init__(self, source: str):
        assert Path(source).exists(), f"File {source} does not exist"
        assert Path(source).is_file(), f"File {source} is not a file"

        self.source = Path(source)
        suffix = self.source.suffix

        if suffix in ['.tiff', '.tif']:
            self.image = rasterio_to_cv2(str(self.source))
            with rasterio.open(str(self.source)) as src:
                self.affine = src.meta['transform']
                self.crs = src.meta['crs']

        elif suffix in ['.jpg', '.jpeg', '.png']:
            self.image = cv2.imread(str(self.source), cv2.IMREAD_COLOR)
            self.affine = None

    def split_channels(self):
        return cv2.split(self.image)


def stepped_rolling_window(arr, win_shape, step=(1, 1)):
    """Applica una rolling window con passo su un array 2D.

    Args:
        arr (np.ndarray): Array 2D di input.
        win_shape (tuple): Dimensione della finestra (win_h, win_w).
        step (tuple): Passo della finestra (step_h, step_w).

    Returns:
        np.ndarray: Array 4D con viste delle finestre (n_rows, n_cols, win_h, win_w).
    """
    win_h, win_w = win_shape
    step_h, step_w = step
    arr_h, arr_w = arr.shape

    # Calcoliamo il numero di finestre lungo ogni dimensione
    out_shape = ((arr_h - win_h) // step_h + 1,
                 (arr_w - win_w) // step_w + 1, win_h, win_w)

    # Creiamo le strided views
    strides = (arr.strides[0] * step_h, arr.strides[1]
               * step_w, arr.strides[0], arr.strides[1])

    return as_strided(arr, shape=out_shape, strides=strides)


@unique
class OPTFLOW_Flags(Enum):
    OPTFLOW_DEFAULT = None
    OPTFLOW_USE_INITIAL_FLOW = 4  # cv2.OPTFLOW_USE_INITIAL_FLOW
    OPTFLOW_FARNEBACK_GAUSSIAN = 256  # cv2.OPTFLOW_FARNEBACK_GAUSSIAN


# definisco una classe con argomenti di default (__init__)
class OpenCVOpticalFlow(OTAlgorithm):
    """
    Wrap-up per la funzione `calcOpticalFlowFarneback` di OpenCV.

    Restituisce l'optical flow tra due immagini

    NOTE: algoritmo di Gunnar Farneback

    Args:
        flow (np.ndarray, optional): computed flow image that has the same size as prev and type CV_32FC2. Defaults to False.
        pyr_scale (float, optional): specifying the image scale to build pyramids for each image. A pyr_scale = 0.5 means a classical pyramid, where each next layer is twice smaller than the previous one. Defaults to 0.5.
        levels (int, optional): number of pyramid layers including the initial image. Setting levels=1 means that no extra layers are created and only the original images are used. Defaults to 4.
        winsize (int, optional): averaging window size. Larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field. Defaults to 16.
        iterations (int, optional): number of iterations the algorithm does at each pyramid level. Defaults to 5.
        poly_n (int, optional): size of the pixel neighborhood used to find polynomial expansion in each pixel. Larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field, typically poly_n=5 or 7. Defaults to 5.
        poly_sigma (float, optional): standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion. For poly_n=5, you can set poly_sigma=1.1, for poly_n=7 a good value would be poly_sigma=1.5. Defaults to 1.1.
        flags (typing.Any, optional): flags operation flags (see `OPTFLOW_Flags`class) that can be a combination of the following:

            - `OPTFLOW_Flags.OPTFLOW_USE_INITIAL_FLOW` (4) uses the input flow as an initial flow approximation.
            - `OPTFLOW_Flags.OPTFLOW_FARNEBACK_GAUSSIAN` (256) uses the Gaussian filter instead of a box filter of the same size for optical flow estimation. Usually, this option gives more accurate flow than with a box filter, at the cost of lower speed. Defaults to OPTFLOW_Flags.OPTFLOW_FARNEBACK_GAUSSIAN.
    """

    def __init__(self, band: int | list[int] = 1, flow: np.ndarray = False,
                 pyr_scale: float = 0.5, levels: int = 4, winsize: int = 16,
                 iterations: int = 5, poly_n: int = 5, poly_sigma: float = 1.1,
                 flags: int = OPTFLOW_Flags.OPTFLOW_DEFAULT) -> None:

        self.band = band
        self.flow = flow
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flags = flags

    @staticmethod
    def from_dict(__d: dict):
        """
        restituisce un'istanza della classe estraendo da `__d` solo gli
        argomenti necessari.
        """
        keys = [
            "flow",
            "pyr_scale",
            "levels",
            "winsize",
            "iterations",
            "poly_n",
            "poly_sigma",
            "flags",
        ]

        kw = dict()
        for key in keys:
            kw[key] = __d.get(key, None)

        return OpenCVOpticalFlow(**kw)

    @staticmethod
    def from_JSON(__json: Path | str):
        __d = json.loads(Path(__json).read_text())

        return OpenCVOpticalFlow.from_dict(__d)

    @staticmethod
    def from_YAML(__yaml: Path | str):
        import yaml

        __d = yaml.safe_load(Path(__yaml).read_text())

        return OpenCVOpticalFlow.from_dict(__d)

    def toJSON(self):

        try:
            parms = self.__dict__
            parms['flags'] = self.flags.value
        except AttributeError:
            parms = self.__dict__

        return json.dumps(parms, indent=4)

    def _to_displacements(self, meta: dict, pixel_offsets) -> tuple:

        px, py = pixel_offsets.T
        dxx, dyy = px.T * meta["transform"].a, py.T * -meta["transform"].e

        return np.linalg.norm([dxx, dyy], axis=0)

    def __call__(self, meta: dict, reference: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Args:
            meta (dict): metadati degli array di input
            reference (np.ndarray): first 8-bit single-channel input image.
            target (np.ndarray): second input image of the same size and the same type as `reference`

        Returns:
            np.ndarray: spostamenti (frazioni di pixel)
        """

        offsets = cv2.calcOpticalFlowFarneback(prev=reference, next=target,
                                               flow=self.flow,
                                               pyr_scale=self.pyr_scale,
                                               levels=self.levels,
                                               winsize=self.winsize,
                                               iterations=self.iterations,
                                               poly_n=self.poly_n,
                                               poly_sigma=self.poly_sigma,
                                               flags=self.flags)

        return self._to_displacements(meta, offsets)


class SkiOpticalFlowILV(OTAlgorithm):
    def __init__(self, band: int | list[int] = 1, radius=7,
                 num_warp=10, gaussian=False, prefilter=False,
                 dtype=np.float32):

        self.band = band
        self.radius = radius
        self.num_warp = num_warp
        self.gaussian = gaussian
        self.prefilter = prefilter
        self.dtype = dtype

    @staticmethod
    def from_dict(__d: dict):
        """
        restituisce un'istanza della classe estraendo da `__d` solo gli
        argomenti necessari.
        """
        keys = [
            "radius",
            "num_warp",
            "gaussian",
            "prefilter",
            "dtype"
        ]

        kw = dict()
        for key in keys:
            kw[key] = __d.get(key, None)

        return SkiOpticalFlowILV(**kw)

    @staticmethod
    def from_JSON(__json: Path | str):
        __d = json.loads(Path(__json).read_text())

        return SkiOpticalFlowILV.from_dict(__d)

    @staticmethod
    def from_YAML(__yaml: Path | str):
        import yaml

        __d = yaml.safe_load(Path(__yaml).read_text())

        return SkiOpticalFlowILV.from_dict(__d)

    def _to_displacements(self, meta: dict, pixel_offsets) -> tuple:
        px, py = pixel_offsets
        dxx, dyy = px * meta["transform"].a, py * -meta["transform"].e

        return np.linalg.norm([dxx, dyy], axis=0)

    def __call__(self, meta: dict, reference: np.ndarray, target: np.ndarray) -> np.ndarray:

        pixel_offsets = optical_flow_ilk(reference, target, radius=self.radius,
                                         num_warp=self.num_warp, gaussian=self.gaussian,
                                         prefilter=self.prefilter, dtype=self.dtype)

        return self._to_displacements(meta, pixel_offsets)


class SkiOpticalFlowTVL1(OTAlgorithm):
    def __init__(self, band=1, attachment=15, tightness=0.3,
                 num_warp=5, num_iter=10, tol=1e-4, prefilter=False,
                 dtype=np.float32):

        self.band = band
        self.attachment = attachment
        self.tightness = tightness
        self.num_warp = num_warp
        self.num_iter = num_iter
        self.tol = tol
        self.prefilter = prefilter
        self.dtype = dtype

    @staticmethod
    def from_dict(__d: dict):
        """
        restituisce un'istanza della classe estraendo da `__d` solo gli
        argomenti necessari.
        """
        keys = [
            "attachment",
            "tightness",
            "num_warp",
            "num_iter",
            "tol",
            "prefilter",
            "dtype"
        ]

        kw = dict()
        for key in keys:
            kw[key] = __d.get(key, None)

        return SkiOpticalFlowTVL1(**kw)

    @staticmethod
    def from_JSON(__json: Path | str):
        __d = json.loads(Path(__json).read_text())

        return SkiOpticalFlowTVL1.from_dict(__d)

    @staticmethod
    def from_YAML(__yaml: Path | str):
        import yaml

        __d = yaml.safe_load(Path(__yaml).read_text())

        return SkiOpticalFlowTVL1.from_dict(__d)

    def _to_displacements(self, meta: dict, pixel_offsets) -> tuple:
        px, py = pixel_offsets
        dxx, dyy = px * meta["transform"].a, py * -meta["transform"].e

        return np.linalg.norm([dxx, dyy], axis=0)

    def __call__(self, meta: dict, reference: np.ndarray, target: np.ndarray) -> np.ndarray:

        pixel_offsets = optical_flow_tvl1(
            reference, target,
            attachment=self.attachment,
            tightness=self.tightness,
            num_warp=self.num_warp,
            num_iter=self.num_iter,
            tol=self.tol,
            prefilter=self.prefilter,
            dtype=self.dtype)

        return self._to_displacements(meta, pixel_offsets)


def process_image(image: np.ndarray, win_size: tuple, step_size: tuple) -> np.ndarray:
    return stepped_rolling_window(image, win_size, step_size).reshape(-1, *win_size)


def xcorr_to_geopandas(ref: Image, tar: Image,
                       win_size: tuple[int] | int,
                       step_size: tuple[int] | int,
                       normalization: str | None = 'phase',
                       upsample_factor: int | float = 1.0) -> gpd.GeoDataFrame:
    """
    Cross-correlazione di immagini. Di default, normalizza le immagini attraverso FFT,
    (eseguendo una cross-correlazione di fase o PCC); assegnando `None` alla variabile
    `normalization` si esegue una cross-correlazione NON normalizzata.

    Arguments:

        ref, tar (Image): immagini di reference e target (geotiff)
        win_size (tuple[int] | int): dimensione finestra mobile. Se è pari a un numero
        intero (n), verrà assunta una finestra mobile quadrata (nxn)
        step_size (tuple[int] | int): intervallo di campionamento. L'immagine verrà 
        scomposta in tante finestre mobili centrate su punti dell'immagine distanziati di
        `step_size` pixels.
        normalization (str | None): tipo di normalizzazione, `phase` per PCC, `None` per
        CC non normalizzata. Default = `phase`
        upsample_factor (int | float): rapporto di sovracampionamento (?). Le immagini
        verranno registrate con una dimensione dei pixel pari a `1 / upsample_factor`.
        Utile per identificare spostamenti a scala di sub-pixel. Influenza molto il carico
        di calcolo necessario.

    Returns:

        (geopandas.GeoDataFrame): shapefile di output. Vengono memorizzati gli attributi di
        spostamento risultante (L2) e spostamento lungo le righe (RSHIFT) e colonne (CSHIFT).
        Tutti gli spostamenti sono espressi nell'unità di misura propria delle immagini di
        partenza.

    """

    if isinstance(win_size, int):
        win_size = win_size, win_size

    if isinstance(step_size, int):
        step_size = step_size, step_size

    _ref = process_image(ref.image, win_size, step_size)
    _tar = process_image(tar.image, win_size, step_size)

    i = np.arange(step_size[0], ref.image.shape[0], step_size[0]).astype(int)
    j = np.arange(step_size[1], ref.image.shape[1], step_size[1]).astype(int)
    index = np.array(list(product(i, j)))

    offset_record = list()

    for (e, r, t) in zip(index, _ref, _tar):
        (sr, sc), _, _ = phase_cross_correlation(
            r, t, normalization=normalization, upsample_factor=upsample_factor)
        l2 = np.sqrt(sr**2 + sc**2)
        offset_record.append(tuple([l2, float(sr), float(sc)]))

    transfomer = AffineTransformer(ref.affine)
    df = pd.DataFrame(offset_record, columns=[
                      "L2", "RSHIFT", "CSHIFT"]) * ref.affine.a  # pixel -> metri
    coords = np.c_[transfomer.xy(*index.T)]
    geom = points_from_xy(*zip(*coords), crs=ref.crs.to_string())

    return gpd.GeoDataFrame(df, geometry=geom)  # .to_file("corvara.shp")
