import json
from enum import Enum, unique
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject
from skimage.registration import optical_flow_ilk, optical_flow_tvl1, phase_cross_correlation
from ot.interfaces import OTAlgorithm

# potrei evitare di usarla


def write_raster(fpath: str | Path, band: int, array: np.ndarray, metadata: dict) -> None:
    with rasterio.open(fp=str(fpath), mode='w', **metadata) as dst:
        dst.write_band(band, array)


@unique
class OPTFLOW_Flags(Enum):
    OPTFLOW_DEFAULT = None
    OPTFLOW_USE_INITIAL_FLOW = 4  # cv2.OPTFLOW_USE_INITIAL_FLOW
    OPTFLOW_FARNEBACK_GAUSSIAN = 256  # cv2.OPTFLOW_FARNEBACK_GAUSSIAN


def basic_pixel_coregistration(infile: str, match: str, outfile: str) -> None:
    """
    Allineamento pixel tra un'immagine target (infile) e un'immagine di reference
    (match). Eventualmente, riproietta sullo stesso CRS di 'match'

    NON RESTITUISCE UN ARRAY MA SCRIVE DIRETTAMENTE IL RASTER COREGISTRATO
    """
    with rasterio.open(infile) as src:
        src_transform = src.transform

        with rasterio.open(match) as match:
            dst_crs = match.crs

            # calcolo trasformazione affine di output
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs,
                dst_crs,
                match.width,
                match.height,
                *match.bounds,  # (left, bottom, right, top)
            )

        # imposta metadati per output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update({"crs": dst_crs,
                           "transform": dst_transform,
                           "width": dst_width,
                           "height": dst_height,
                           "nodata": 0})

        print("Coregistrato a dimensioni:", dst_height,
              dst_width, '\n Affine\n', dst_transform)

        # Output
        with rasterio.open(outfile, "w", **dst_kwargs) as dst:
            # itero tutte le bande di 'infile'
            for i in range(1, src.count + 1):
                reproject(source=rasterio.band(src, i),
                          destination=rasterio.band(dst, i),
                          src_transform=src.transform,
                          src_crs=src.crs,
                          dst_transform=dst_transform,
                          dst_crs=dst_crs,
                          resampling=Resampling.bilinear)


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

    def __init__(self, flow: np.ndarray = False, pyr_scale: float = 0.5,
                 levels: int = 4, winsize: int = 16, iterations: int = 5,
                 poly_n: int = 5, poly_sigma: float = 1.1,
                 flags: int = OPTFLOW_Flags.OPTFLOW_DEFAULT) -> None:

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

    @classmethod
    def run(cls, **kwargs) -> np.ndarray:  # potrebbe stare da sola

        otalg = cls.from_dict(kwargs)
        coreg_output = Path(kwargs['target']).parent / \
            (Path(kwargs['target']).stem+'_coreg.tif')

        basic_pixel_coregistration(
            kwargs['target'], kwargs['reference'], coreg_output)

        reference = rasterio.open(
            kwargs['reference']).read(2).astype(np.float32)
        target = rasterio.open(coreg_output).read(2).astype(np.float32)
        mask = ~rasterio.open(coreg_output).dataset_mask().astype(bool)
        meta = rasterio.open(coreg_output).meta

        resdispl = otalg(meta, reference, target)

        meta['dtype'] = 'float32'
        meta['count'] = 1
        resdispl_ma = np.ma.masked_array(resdispl, mask)
        # normalizzo (o no?)
        # resdispl_ma = (resdispl_ma - np.mean(resdispl_ma))/np.std(resdispl_ma)
        print(np.min(resdispl_ma))
        print(np.max(resdispl_ma))

        write_raster(kwargs["output"], 1, resdispl_ma, meta)


class SkiOpticalFlowILV(OTAlgorithm):
    def __init__(self, radius=7, num_warp=10, gaussian=False,
                 prefilter=False, dtype=np.float32):

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

    @classmethod
    def run(cls, **kwargs) -> np.ndarray:  # potrebbe stare da sola

        otalg = cls.from_dict(kwargs)
        coreg_output = Path(kwargs['target']).parent / \
            (Path(kwargs['target']).stem+'_coreg.tif')

        basic_pixel_coregistration(
            kwargs['target'], kwargs['reference'], coreg_output)

        reference = rasterio.open(
            kwargs['reference']).read(2).astype(np.float32)
        target = rasterio.open(coreg_output).read(2).astype(np.float32)
        mask = ~rasterio.open(coreg_output).dataset_mask().astype(bool)
        meta = rasterio.open(coreg_output).meta

        resdispl = otalg(meta, reference, target)

        meta['dtype'] = 'float32'
        meta['count'] = 1
        resdispl_ma = np.ma.masked_array(resdispl, mask)
        # normalizzo (o no?)
        # resdispl_ma = (resdispl_ma - np.mean(resdispl_ma))/np.std(resdispl_ma)
        print(np.min(resdispl_ma))
        print(np.max(resdispl_ma))

        write_raster(kwargs["output"], 1, resdispl_ma, meta)


class SkiOpticalFlowTVL1(OTAlgorithm):
    def __init__(self, attachment=15, tightness=0.3, num_warp=5, num_iter = 10,
                 tol=1e-4, prefilter=False, dtype=np.float32):
        
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
            attachment = self.attachment,
            tightness = self.tightness,
            num_warp = self.num_warp,
            num_iter = self.num_iter,
            tol = self.tol,
            prefilter = self.prefilter,
            dtype = self.dtype)

        return self._to_displacements(meta, pixel_offsets)

    @classmethod
    def run(cls, **kwargs) -> np.ndarray:  # potrebbe stare da sola

        otalg = cls.from_dict(kwargs)
        coreg_output = Path(kwargs['target']).parent / \
            (Path(kwargs['target']).stem+'_coreg.tif')

        basic_pixel_coregistration(
            kwargs['target'], kwargs['reference'], coreg_output)

        reference = rasterio.open(
            kwargs['reference']).read(2).astype(np.float32)
        target = rasterio.open(coreg_output).read(2).astype(np.float32)
        mask = ~rasterio.open(coreg_output).dataset_mask().astype(bool)
        meta = rasterio.open(coreg_output).meta

        resdispl = otalg(meta, reference, target)

        meta['dtype'] = 'float32'
        meta['count'] = 1
        resdispl_ma = np.ma.masked_array(resdispl, mask)
        # normalizzo (o no?)
        # resdispl_ma = (resdispl_ma - np.mean(resdispl_ma))/np.std(resdispl_ma)
        print(np.min(resdispl_ma))
        print(np.max(resdispl_ma))

        write_raster(kwargs["output"], 1, resdispl_ma, meta)
