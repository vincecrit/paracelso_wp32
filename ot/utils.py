import logging
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject

logger = logging.getLogger(__name__)


def basic_pixel_coregistration(infile: str, match: str,
                               outfile: str | None = None) -> Path:
    """Align pixels between a target image (infile) and a reference image (match).
    Optionally, reproject to the same CRS as 'match'."""
    
    if outfile is None:
        out_stem = Path(infile).stem + "_coreg" + Path(infile).suffix
        outfile = Path(infile).parent / out_stem
    
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
        
        return outfile
    

def is_identity_affine(affine: rasterio.Affine) -> bool:
    matrix = np.array(affine).reshape(3, 3)
    if (np.diagonal(matrix) == 1).all():
        return True
    else:
        return False


def rasterio_open(source: str, band: int | None = None) -> np.ndarray:
    logger.info("Caricamento dataset raster")
    logger.debug(f"Caricamento {source} con rasterio.")

    with rasterio.open(source) as src:
        if band is None:
            iter_bands = range(src.count)
        elif band > (src.count - 1):
            logging.critical(f"La banda selezionata non esiste. "+
                                f"Numero bande dataset: {src.count}. "+
                                "Gli indici delle bande partono da zero")
            exit(0)
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
