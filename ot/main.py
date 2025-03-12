import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from ot import logger
from ot.helpmsg import (ALGNAME, ATTACHMENT, BAND, CLAHE, FLAGS, FLOW,
                        GAUSSIAN, ITERATIONS, LEVELS, LOGNORM, MINMAX, NODATA,
                        NORMALIZE, NUMITER, NUMWARP, OUTPUT, PHASENORM, POLY_N,
                        POLY_SIGMA, PREFILTER, PYR_SCALE, RADIUS, REFERENCE,
                        STEPSIZE, TARGET, TIGHTNESS, TOL, UPSAMPLE_FACTOR,
                        WINSIZE, ZSCORENORM)
from ot.image_processing import dispatcher
from ot.interfaces import Image
from ot.metodi import get_method
from ot.utils import (DriverCapabilityError, RasterioIOError,
                      basic_pixel_coregistration, geopandas_to_gpkg, _is_identity_affine,
                      rasterio_read, image_to_raster)


def get_parser() -> argparse.ArgumentParser:  # cosa faccio con questo mostro?
    """Get the argument parser for the script."""
    parser = argparse.ArgumentParser(
        description="Offset-Tracking PARACELSO WP3.2")

    # parametri comuni
    parser.add_argument("-ot", "--algname", help=ALGNAME, type=str)
    parser.add_argument("-r", "--reference", help=REFERENCE, type=str)
    parser.add_argument("-t", "--target", help=TARGET, type=str)
    parser.add_argument("-o", "--output", help=OUTPUT,
                        default="output.tif", type=str)
    parser.add_argument("-b", "--band", help=BAND, default=None, type=str)
    parser.add_argument("--nodata", help=NODATA, default=None, type=float)
    parser.add_argument("-prep", "--preprocessing", default=None, type=str)
    parser.add_argument("--out_format", default=None, type=str)

    # parametri OpenCV
    parser.add_argument("--flow", help=FLOW, default=None)
    parser.add_argument("--levels", help=LEVELS, default=4, type=int)
    parser.add_argument("--pyr_scale", help=PYR_SCALE, default=0.5, type=float)
    parser.add_argument("--winsize", help=WINSIZE, type=int, default=4)
    parser.add_argument("--step_size", help=STEPSIZE, type=int, default=1)
    parser.add_argument("--iterations", help=ITERATIONS, default=10, type=int)
    parser.add_argument("--poly_n", help=POLY_N, default=5, type=int)
    parser.add_argument("--poly_sigma", help=POLY_SIGMA,
                        default=1.1, type=float)
    parser.add_argument("--flags", help=FLAGS, default=None, type=int)

    # parametri scikit-image
    parser.add_argument("--radius", help=RADIUS, type=int, default=4)
    parser.add_argument("--num_warp", help=NUMWARP, type=int, default=3)
    parser.add_argument("--gaussian", help=GAUSSIAN, action="store_true")
    parser.add_argument("--prefilter", help=PREFILTER, action="store_true")
    parser.add_argument("--attachment", help=ATTACHMENT, type=int, default=10)
    parser.add_argument("--tightness", help=TIGHTNESS, type=float, default=0.3)
    parser.add_argument("--num_iter", help=NUMITER, type=int, default=10)
    parser.add_argument("--tol", help=TOL, type=float, default=1e-4)
    parser.add_argument("--phase_norm", help=PHASENORM,
                        action="store_true", default=True)
    parser.add_argument("--upsmp_fac", help=UPSAMPLE_FACTOR,
                        type=float, default=1.0)

    return parser


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
        case _:
            raise NotImplementedError
        

def load_file(source, **kwargs):
    # forse si può fare a meno di averla, è diventata un
    #  wrap-up di `ot.utils.rasterio_open`
    source = Path(source)

    if not source.is_file():
        logger.critical(f"Il file {source} non esiste")
        exit(0)

    return Image(*rasterio_read(source, **kwargs))


def load_images(*args):
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
        reference = load_file(reference_file)
        target = load_file(target_file)

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


def _summary_statistics(array):  # For fun
    logger.debug(f"{'OT Media':>22s}: {np.mean(array): .3g}")
    logger.debug(f"{'OT Mediana':>22s}: {np.median(array): .3g}")
    logger.debug(f"{'OT STD':>22s}: {np.std(array): .3g}")
    logger.debug(f"{'OT Minimo':>22s}: {np.min(array): .3g}")
    logger.debug(f"{'OT Massimo':>22s}: {np.max(array): .3g}")
    logger.debug(f"{'OT 25° perc.':>22s}: {np.percentile(array, 25): .3g}")
    logger.debug(f"{'OT 75° perc.':>22s}: {np.percentile(array, 75): .3g}")
    

def main() -> None:
    args = get_parser().parse_args()
    algorithm = get_method(args.algname).from_dict(vars(args))
    algorithm.toJSON()
    logger.info(f"AVVIO ANALISI OT CON METODO {algorithm.__class__.__name__}")

    reference, target = load_images(args.reference, args.target)
    preprocessed_images = [
        dispatcher.dispatch_process(f"{algorithm.library}_{args.preprocessing}", array=img)
        for img in (reference, target)]

    logger.info(f"{args.preprocessing.upper()} eseguito correttamente.")
    displacements = algorithm(*preprocessed_images)
    logger.info(f"Algoritmo {algorithm.__class__.__name__} eseguito correttamente")

    _summary_statistics(displacements)

    try:
        logger.info(f"Esporto su file: {args.output}")
        write_output(displacements, args.output)

    except (ValueError, TypeError, RasterioIOError,
            DriverCapabilityError, NotImplementedError) as err:
        logger.critical(f"[RASTERIO]{err.__class__.__name__}: {err}")
        exit(0)


if __name__ == "__main__":
    main()
