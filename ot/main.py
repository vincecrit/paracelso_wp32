import argparse
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
from ot.utils import basic_pixel_coregistration, cv2imread, load_raster


def get_parser() -> argparse.ArgumentParser:
    """Get the argument parser for the script."""
    parser = argparse.ArgumentParser(description="Optical flow")

    # parametri comuni
    parser.add_argument("-ot", "--algname", help=ALGNAME, type=str)
    parser.add_argument("-r", "--reference", help=REFERENCE, type=str)
    parser.add_argument("-t", "--target", help=TARGET, type=str)
    parser.add_argument("-o", "--output", help=OUTPUT,
                        default="output.tif", type=str)
    parser.add_argument("-b", "--band", help=BAND, default=None, type=str)
    parser.add_argument("--nodata", help=NODATA, default=None, type=float)
    parser.add_argument("--preprocessing", default=None, type=str)

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

    parser

    return parser


def load_file(source, **kwargs):
    source = Path(source)

    if not source.is_file():
        logger.critical(f"Il file {source} non esiste")
        exit(0)

    match source.suffix:
        case ".tif":
            dataset, affine, crs = load_raster(source, **kwargs)
            img_format = "raster"
        case ".tiff":
            dataset, affine, crs = load_raster(source, **kwargs)
            img_format = "raster"
        case ".jpg":
            dataset = cv2imread(source, cv2.COLOR_BGR2RGB)
            affine = None
            crs = None
            img_format = "generic"
        case ".jpeg":
            dataset = cv2imread(source, cv2.COLOR_BGR2RGB)
            affine = None
            crs = None
            img_format = "generic"
        case ".png":
            dataset = cv2imread(source, cv2.COLOR_BGR2RGB)
            affine = None
            crs = None
            img_format = "generic"

    return img_format, Image(dataset, affine, crs)


def load_images(*args):
    try:
        reference_file, target_file = [Path(src) for src in args]

    except ValueError as err:
        logger.critical(f"Troppi argomenti di input (attesi 2)")
        logger.debug(f"{err}")
        exit(0)

    logger.info(f"REFERENCE: {reference_file.name}")
    logger.info(f"TARGET: {target_file.name}")

    if not reference_file.suffix == target_file.suffix:
        logger.critical("Le due immagini hanno formati diversi")
        logger.debug(f"{reference_file.suffix=}, {target_file.suffix=}")
        exit(0)

    else:
        ref_format, reference = load_file(reference_file)
        tar_format, target = load_file(target_file)

        if ref_format == tar_format == "raster":
            if not reference.is_coregistered(target):
                logger.info("Eseguo coregistrazione" +
                            "tra immagini raster")
                target_coreg = target_file.parent / (target_file.stem +
                                                     "_coreg" + 
                                                     target_file.suffix)
                basic_pixel_coregistration(str(target_file), str(reference_file),
                                           str(target_coreg))
                logger.info("Coregistrazione eseguita correttamente." +
                            f" File coregistrato: {target_coreg}")
                
                _, target = load_file(target_coreg)
            else:
                logger.info("Immagini raster già coregistrate.")
        else:
            logger.info("Immagini di input non raster")
            if not reference.is_coregistered(target):
                logger.warning("La coregistrazione tra immagini non georiferite ")
            else:
                logger.info("Le immagini di input presentano le medesime dimensioni")
        return reference, target


def _summary_statistics(array):
    logger.debug(f"{'OT Media':>22s}: {np.mean(array): .3g}")
    logger.debug(f"{'OT Mediana':>22s}: {np.median(array): .3g}")
    logger.debug(f"{'OT STD':>22s}: {np.std(array): .3g}")
    logger.debug(f"{'OT Minimo':>22s}: {np.min(array): .3g}")
    logger.debug(f"{'OT Massimo':>22s}: {np.max(array): .3g}")
    logger.debug(f"{'OT 25° perc.':>22s}: {np.percentile(array, 25): .3g}")
    logger.debug(f"{'OT 75° perc.':>22s}: {np.percentile(array, 75): .3g}")


def main() -> None:
    args = get_parser().parse_args()

    method = get_method(args.algname)
    algorithm = method.from_dict(vars(args))

    logger.info(f"AVVIO ANALISI OT CON METODO {algorithm.__class__.__name__}")

    algorithm.toJSON()  # salvo parametri utilizzati

    reference, target = load_images(args.reference, args.target)

    prep_imgs = list()
    for img in (reference, target):
        match algorithm.library:
            case "OpenCV":
                info_ = f"Applicazione {args.preprocessing.upper()} mediante libreria OpenCV"
                logger.info(info_)

                output = dispatcher.dispatch_process(
                    "cv2_"+args.preprocessing, array=img)
                prep_imgs.append(output)

            case "scikit-image":
                info_ = f"Applicazione {args.preprocessing.upper()} mediante libreria scikit-image"
                logger.info(info_)

                output = dispatcher.dispatch_process(
                    "ski_"+args.preprocessing, array=img)
                prep_imgs.append(output)

    logger.info(f"{args.preprocessing.upper()} eseguito correttamente.")

    logger.debug(
        f"Output {args.preprocessing.upper()}: {[type(e) for e in prep_imgs]}")
    displacements = algorithm(*prep_imgs)
    logger.info(
        f"Algoritmo {algorithm.__class__.__name__} eseguito correttamente")
    logger.info(f"Esporto su file: {args.output}")

    _summary_statistics(displacements)


if __name__ == "__main__":
    main()
