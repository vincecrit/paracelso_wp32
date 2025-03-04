import logging

import cv2
import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject

from ot.image_processing import opencv, ski

logger = logging.getLogger(__name__)


class PreprocessDispatcher:
    def __init__(self):
        self.processes = dict()

    def register(self, name: str, process):
        if name not in self.processes:
            self.processes[name] = list()
        self.processes[name].append(process)

    def dispatch_process(self, name: str, **kwargs):
        for process in self.processes[name]:
            return process(**kwargs)


def basic_pixel_coregistration(infile: str, match: str, outfile: str) -> None:
    """Align pixels between a target image (infile) and a reference image (match).
    Optionally, reproject to the same CRS as 'match'."""
    with rasterio.open(infile) as src:
        src_transform = src.transform
        nodata = src.meta['nodata']

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
                           "nodata": nodata})

        logger.debug(
            f"Immagine coregistrata a dimensioni: {dst_height}, {dst_width}")

        # Output
        logger.info(f"Esporto immagine: {outfile}")
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


def load_raster(source: str, band: int | None = None) -> np.ndarray:
    logger.debug("Caricamento dataset raster con rasterio")
    with rasterio.open(source) as src:
        if band is None:
            iter_bands = range(src.count)
        else:
            iter_bands = [band]

        channels = []
        for b in iter_bands:
            band = src.read(b+1)
            band[band == src.meta['nodata']] = 0
            channels.append(band)

        dataset = cv2.merge(channels)
        affine = src.meta['transform']
        crs = src.meta['crs']

        return dataset, affine, crs


def cv2imread(*args, **kwargs):
    logger.debug("Caricamento immagini con OpenCV")
    return cv2.imread(*args, **kwargs)


def registration():
    dispatcher = PreprocessDispatcher()

    dispatcher.register("cv2_clahe", opencv.clahe)
    dispatcher.register("ski_clahe", ski.clahe)
    dispatcher.register("cv2_equalize", opencv.equalize)
    dispatcher.register("ski_equalize", ski.equalize)
    dispatcher.register("cv2_lognorm", opencv.lognorm)
    dispatcher.register("ski_lognorm", ski.lognorm)

    return dispatcher
