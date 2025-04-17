import logging
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject

from log import setup_logger
from ot.interfaces import Image


logger = setup_logger(__name__)


def write_output(output, outfile: str | Path) -> None:

    outfile = Path(outfile)

    match outfile.suffix:
        case ".tiff":
            image_to_raster(output, outfile)
        case ".tif":
            image_to_raster(output, outfile)
        case ".jpg":
            raise NotImplementedError
        case ".jpeg":
            raise NotImplementedError
        case ".png":
            raise NotImplementedError
        case ".gpkg":
            geopandas_to_gpkg(output, outfile)
        case ".shp":
            geopandas_to_gpkg(output, outfile)
        case _:
            raise NotImplementedError(f"Unsupported file extension: {outfile.suffix}")


def load_file(source, **kwargs):
    # forse si può fare a meno di averla, è diventata un
    #  wrap-up di `ot.utils.rasterio_open`
    source = Path(source)

    if not source.is_file():
        logger.critical(f"Il file {source} non esiste")
        exit(0)

    return Image(*rasterio_read(source, **kwargs))


def load_images(*args, band):
    """
    Carica una coppia di immagini e resituisce una coppia di oggetti 
    `ot.interfaces.Image`.

    Info:
        Operazioni eseguite:
        1. Controllo estensione dei file
        2. Controllo georeferenziazione
        3. Coregistrazione di immagini
        4. Output (oggetti `ot.interfaces.Image`)

    Args:
        *args(str): percorsi delle immagini di reference e target, in quest'ordine

    Returns:
        tuple(ot.interfaces.Image):
    """
    #  [1] Creo oggetti Path
    reference_file, target_file = [Path(src) for src in args]
    logger.info(f"REFERENCE: {reference_file.name}")
    logger.info(f"TARGET: {target_file.name}")

    # file con estensione diversa non sono accettati
    if not reference_file.suffix == target_file.suffix:
        logger.critical("Le due immagini hanno formati diversi")
        logger.debug(f"{reference_file.suffix=}, {target_file.suffix=}")
        exit(0)

    else:
        # carico qualsiasi tipo di file con rasterio (tanto legge tutto mwaahahah)
        reference = load_file(reference_file, band=band)
        target = load_file(target_file, band=band)

        # [2] rasterio associa un oggetto Affine come matrice identità quando la
        # georeferenziazione non è definita
        are_identity_affines = [
            _is_identity_affine(e.affine) for e in (reference, target)
        ]

        # Se nessuno dei file è georiferito non dovrebbero esserci problemi (credo)
        if all(are_identity_affines):
            logger.warning("Nessuno dei due file possiede una georeferenziazione. " +
                           "La coregistrazione si limiterà ad allinerare/scalare i pixel " +
                           "dell'immagine target")
            pass

        # Se la georeferenziazione è definita solo per uno dei due file, non
        # so che fare... Quindi lo butto fuori.
        elif any(are_identity_affines):
            logger.critical(
                "Uno dei due file non possiede la georeferenziazione.")
            logger.debug(f"{_is_identity_affine(reference.affine)=}")
            logger.debug(f"{_is_identity_affine(target.affine)=}")
            exit(0)

        # [3] Coregistrazione delle immagini
        else:
            if not reference.is_coregistered(target):
                logger.info("Eseguo coregistrazione tra immagini raster")
                target_coreg = basic_pixel_coregistration(str(target_file),
                                                          str(reference_file))
                logger.info(f"Coregistrazione eseguita correttamente" +
                            f"File coregistrato: {target_coreg}")
                target = load_file(target_coreg)
            else:
                # non succede mai in realtà
                logger.info("Immagini raster già coregistrate.")

        return reference, target


def _to_bandlast(arr):
    '''[BAND, ROW, COL] -> [ROW, COL, BAND]'''
    return np.transpose(arr, (1, 2, 0))


def __debug_attrerr(func, *args, **kwargs):
    def inner(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except AttributeError as err:
            logger.critical(f"{err.__class__.__name__}: {err.__str__()}")
        return None
    return inner


def _is_image(arg) -> bool:
    return isinstance(arg, Image)


def _is_geodf(arg) -> bool:
    return isinstance(arg, gpd.GeoDataFrame)


def _is_identity_affine(affine: rasterio.Affine) -> bool:
    matrix = np.array(affine).reshape(3, 3)
    if (np.diagonal(matrix) == 1).all():
        return True
    else:
        return False


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


def rasterio_read(source: str, band: int | None = None) -> np.ndarray:
    logger.debug(f"Caricamento {source} con rasterio.")

    with rasterio.open(source) as src:

        if band is None:
            iter_bands = range(min(3, src.count))

        elif band > (src.count - 1):
            logging.critical(f"La banda selezionata non esiste. " +
                             f"Numero bande dataset: {src.count}. " +
                             "Gli indici delle bande partono da zero")
            exit(0)

        else:
            iter_bands = [band]

        channels = []
        for b in iter_bands:
            band = src.read(b+1)
            channels.append(band)

    dataset = cv2.merge(channels)
    affine = src.meta['transform']
    crs = src.meta['crs']

    return dataset, affine, crs


def image_to_raster(img: Image, outfile) -> None:

    if not _is_image(img):
        raise ValueError("Tipo di argomento non corretto. " +
                         f"Atteso `{type(Image)}`, ricevuto `{type(img)}`")

    with rasterio.open(outfile, "w", transform=img.affine, crs=img.crs, nodata=0,
                       width=img.width, height=img.height, dtype=img.image.dtype, count=1) as ds:
        ds.write(img.image, 1)


def geopandas_to_gpkg(frame, outfile) -> None:

    if not _is_geodf(frame):
        raise ValueError("Tipo di argomento non corretto. " +
                         f"Atteso `{type(gpd.GeoDataFrame)}`, ricevuto `{type(frame)}`")

    frame.to_file(outfile, layer="spostamenti")
