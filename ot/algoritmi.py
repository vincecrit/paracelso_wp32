"""
Modulo per l'elaborazione di immagini e la registrazione tramite tecniche di optical flow e cross-correlazione di fase.
Questo modulo contiene diverse classi e funzioni per calcolare il flusso ottico tra immagini e per eseguire la cross-correlazione di fase tra immagini di riferimento e target. Le classi forniscono un'interfaccia per utilizzare le funzioni di optical flow di OpenCV e scikit-image, mentre le funzioni permettono di convertire i risultati in formati utilizzabili come GeoDataFrame di geopandas o DataFrame di pandas.
Classi:
    - OpenCVOpticalFlow: Wrap-up per la funzione `calcOpticalFlowFarneback` di OpenCV.
    - SkiOpticalFlowILK: Wrapper per la funzione `optical_flow_ilk` di scikit-image.
    - SkiOpticalFlowTVL1: Wrapper per la funzione `optical_flow_tvl1` di scikit-image.
    - SkiPCC_Vector: Wrapper per il calcolo del vettore di cross-correlazione di fase.
Funzioni:
    - stepped_rolling_window: funzione per generazione di finestre mobili.
    - xcorr_to_frame: Esegue la cross-correlazione di immagini e restituisce un GeoDataFrame o DataFrame con i risultati.
"""
import logging
from enum import Enum, unique

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import points_from_xy
from rasterio.transform import AffineTransformer
from skimage.registration import (optical_flow_ilk, optical_flow_tvl1,
                                  phase_cross_correlation)
from skimage.util import view_as_windows
from tqdm import tqdm

from ot.interfaces import Image, OTAlgorithm

logger = logging.getLogger(__name__)


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


# non so dove metterla
def xcorr_to_frame(ref: Image, tar: Image,
                   win_size: tuple[int] | int,
                   step_size: int,
                   normalization: str | None = 'phase',
                   upsample_factor: int | float = 1.0) -> gpd.GeoDataFrame | pd.DataFrame:
    """
    Cross-correlazione di immagini. Di default, normalizza le immagini
    attraverso FFT, eseguendo quindi una cross-correlazione di fase (PCC);
    assegnando `None` alla variabile `normalization` si esegue una
    cross-correlazione NON normalizzata.

    Le due immagini di input vengono elaborate mediante la medesima finestra.

    Args:
        ref, tar (Image): immagini di reference e target (geotiff)
        win_size (tuple[int] | int): dimensione finestra mobile. Se è pari a un
        numero intero (n), verrà assunta una finestra mobile quadrata (nxn)
        step_size (tuple[int] | int): intervallo di campionamento. L'immagine
        verrà scomposta in tante finestre mobili centrate su punti dell'immagine
        distanziati di `step_size` pixels.
        normalization (str | None): tipo di normalizzazione, `phase` per PCC,
        `None` per CC non normalizzata. Default = `phase`
        upsample_factor (int | float): rapporto di sovracampionamento (?).
        Le immagini verranno registrate con una dimensione dei pixel pari a
        `1 / upsample_factor`. Utile per identificare spostamenti a scala di
        sub-pixel. Influenza molto il carico di calcolo necessario.

    Returns:
        frame (geopandas.GeoDataFrame | pandas.DataFrame): shapefile o
        dataframe di output (dipende dal tipo di immagine: raster -> shapefile)
        Vengono memorizzati gli attributi di spostamento risultante (L2) e
        spostamento lungo le righe (RSHIFT) e colonne (CSHIFT). Tutti gli
        spostamenti sono espressi nell'unità di misura propria delle immagini
        di partenza.
    """
        
    logger.info("Eseguo algoritmo skimage.registration.phase_cross_correlation")
    logger.debug(f"{ref.shape = }, {ref.image.dtype = }")
    logger.debug(f"{tar.shape = }, {tar.image.dtype = }")

    if ref.crs is None:
        CRS = None
    else:
        CRS = ref.crs.to_string()

    if isinstance(win_size, int):
        win_size = win_size, win_size

    _ref, index = stepped_rolling_window(ref.image, win_size, step_size)
    assert index.shape[0] == _ref.shape[0], f"{index.shape[0] =} {_ref.shape[0] =}"
    _tar, _ = stepped_rolling_window(tar.image, win_size, step_size)

    offset_record = list()
    for (r, t) in tqdm(iterable=zip(_ref, _tar), desc=f"IMGCORR", total=index.shape[0], ncols=150):
        (sr, sc), _, _ = phase_cross_correlation(r, t,
                                                 normalization=normalization,
                                                 upsample_factor=upsample_factor)

        L2 = np.sqrt(sr**2 + sc**2)  # risultante dello spostamento
        row = L2, float(sr), float(sc)
        offset_record.append(row)

    columns = ["L2", "RSHIFT", "CSHIFT"]
    df = pd.DataFrame(offset_record, columns=columns)

    if ref.affine is not None:  # caso del raster di input
        transfomer = AffineTransformer(ref.affine)
        df *= ref.affine.a  # pixel -> metri
        coords = np.c_[transfomer.xy(*index.T)]
        geom = points_from_xy(*zip(*coords), crs=CRS)

        return gpd.GeoDataFrame(df, geometry=geom)

    else:  # caso di immagini *.jpeg ecc..
        x, y = index.T
        df.insert(0, 'Y', y)
        df.insert(0, 'X', x)
        return df


@unique
class OPTFLOW_Flags(Enum):
    """Flags for OpenCV optical flow."""
    OPTFLOW_DEFAULT = None
    OPTFLOW_USE_INITIAL_FLOW = 4  # cv2.OPTFLOW_USE_INITIAL_FLOW
    OPTFLOW_FARNEBACK_GAUSSIAN = 256  # cv2.OPTFLOW_FARNEBACK_GAUSSIAN


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

    library = 'OpenCV'

    def __init__(self, flow: np.ndarray = None,
                 pyr_scale: float = 0.5, levels: int = 4, winsize: int = 16,
                 iterations: int = 5, poly_n: int = 5, poly_sigma: float = 1.1,
                 flags: int = OPTFLOW_Flags.OPTFLOW_DEFAULT) -> None:

        self.flow = flow
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        try:
            self.flags = flags.value
        except AttributeError:
            self.flags = flags\

    def __call__(self, reference: Image, target: Image) -> Image:
        """
        Calculate optical flow between reference and target images.
        """
        logger.info("Eseguo algoritmo cv2.calcOpticalFlowFarneback")
        logger.debug(f"{reference.shape = }, {reference.image.dtype = }")
        logger.debug(f"{target.shape = }, {target.image.dtype = }")

        pixel_offsets = cv2.calcOpticalFlowFarneback(prev=reference.image, next=target.image,
                                                     flow=self.flow,
                                                     pyr_scale=self.pyr_scale,
                                                     levels=self.levels,
                                                     winsize=self.winsize,
                                                     iterations=self.iterations,
                                                     poly_n=self.poly_n,
                                                     poly_sigma=self.poly_sigma,
                                                     flags=self.flags)

        displ = self._to_displacements(target.affine, pixel_offsets)

        logger.debug(f"Tipo output: {displ.dtype}")
        logger.debug(f"Shape output: {displ.shape}")

        return Image(displ, target.affine, target.crs, target.nodata)


class SkiOpticalFlowILK(OTAlgorithm):
    """
    Wrapper for scikit-image's optical_flow_ilk function.
    """

    library = 'skimage'

    def __init__(self, radius=7,
                 num_warp=10, gaussian=False, prefilter=False):

        self.radius = radius
        self.num_warp = num_warp
        self.gaussian = gaussian
        self.prefilter = prefilter

    def __call__(self, reference: Image, target: Image) -> Image:
        """
        Calculate optical flow between reference and target images using ILK method.
        """
        logger.info("Eseguo algoritmo skimage.registration.optical_flow_ilk")
        logger.debug(f"{reference.shape = }, {reference.image.dtype = }")
        logger.debug(f"{target.shape = }, {target.image.dtype = }")

        pixel_offsets = optical_flow_ilk(reference.image, target.image, radius=self.radius,
                                         num_warp=self.num_warp, gaussian=self.gaussian,
                                         prefilter=self.prefilter)

        logger.debug(f"Numero output: {len(pixel_offsets)}")
        logger.debug(f"Tipo output: {[type(e) for e in pixel_offsets]}")
        logger.debug(f"Shape output: {[e.shape for e in pixel_offsets]}")

        a, b = pixel_offsets
        displ = self._to_displacements(target.affine, cv2.merge([a, b]))

        return Image(displ, target.affine, target.crs, target.nodata)


class SkiOpticalFlowTVL1(OTAlgorithm):
    """
    Wrapper for scikit-image's optical_flow_tvl1 function
    """

    library = 'skimage'

    def __init__(self, attachment=15, tightness=0.3,
                 num_warp=5, num_iter=10, tol=1e-4, prefilter=False):

        self.attachment = attachment
        self.tightness = tightness
        self.num_warp = num_warp
        self.num_iter = num_iter
        self.tol = tol
        self.prefilter = prefilter

    def __call__(self, reference: Image, target: Image) -> Image:
        """
        Calculate optical flow between reference and target images using TVL1 method.
        """
        logger.info("Eseguo algoritmo skimage.registration.optical_flow_tvl1")
        logger.debug(f"{reference.image.shape = }, {reference.image.dtype = }")
        logger.debug(f"{target.image.shape = }, {target.image.dtype = }")

        pixel_offsets = optical_flow_tvl1(
            reference.image, target.image,
            attachment=self.attachment,
            tightness=self.tightness,
            num_warp=self.num_warp,
            num_iter=self.num_iter,
            tol=self.tol,
            prefilter=self.prefilter)

        logger.debug(f"Numero output: {len(pixel_offsets)}")
        logger.debug(f"Tipo output: {[type(e) for e in pixel_offsets]}")
        logger.debug(f"Shape output: {[e.shape for e in pixel_offsets]}")

        a, b = pixel_offsets
        displ = self._to_displacements(target.affine, cv2.merge([a, b]))

        return Image(displ, target.affine, target.crs, target.nodata)


class SkiPCC_Vector(OTAlgorithm):
    """
    Wrapper for phase cross-correlation vector calculation.
    """

    library = 'skimage'

    def __init__(self, winsize: tuple[int] | int,
                 step_size: tuple[int] | int,
                 phase_norm: bool = True,
                 upsmp_fac: int | float = 1.0):

        if phase_norm:
            self.normalization = 'phase'
        else:
            self.normalization = None

        self.winsize = winsize
        self.step_size = step_size
        self.upsmp_fac = upsmp_fac

    def __call__(self, reference: Image, target: Image) -> gpd.GeoDataFrame | pd.DataFrame:
        """Calculate phase cross-correlation between reference and target images."""
        # self.toJSON(self.__dict__)

        return xcorr_to_frame(ref=reference, tar=target,
                              win_size=self.winsize,
                              step_size=self.step_size,
                              normalization=self.normalization,
                              upsample_factor=self.upsmp_fac)
