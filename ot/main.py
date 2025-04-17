import argparse
import logging
from pathlib import Path

import numpy as np

from log import setup_logger
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
                      _is_identity_affine, basic_pixel_coregistration,
                      geopandas_to_gpkg, image_to_raster, load_images,
                      rasterio_read, write_output)


logger = setup_logger(__name__)


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
    parser.add_argument("-b", "--band", help=BAND, default=None, type=int)
    parser.add_argument("--nodata", help=NODATA, default=None, type=float)
    parser.add_argument("-prep", "--preprocessing",
                        default='equalize', type=str)
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


def main() -> None:
    args = get_parser().parse_args()
    if not args.algname or not args.reference or not args.target:
        logger.critical("Missing required arguments: --algname, --reference, --target")
        exit(1)
    algorithm = get_method(args.algname).from_dict(vars(args))
    logger.info(f"AVVIO ANALISI OT CON METODO {algorithm.__class__.__name__}")

    algorithm.toJSON()

    reference, target = load_images(
        args.reference, args.target, band=args.band)
    preprocessed_images = [
        dispatcher.dispatch_process(
            f"{algorithm.library}_{args.preprocessing}", array=img)
        for img in (reference, target)]

    logger.info(f"{args.preprocessing.upper()} eseguito correttamente.")
    displacements = algorithm(*preprocessed_images)
    logger.info(
        f"Algoritmo {algorithm.__class__.__name__} eseguito correttamente")

    try:
        logger.info(f"Esporto su file: {args.output}")
        write_output(displacements, args.output)

    except (ValueError, TypeError, RasterioIOError,
            DriverCapabilityError, NotImplementedError) as err:
        logger.critical(f"[RASTERIO]{err.__class__.__name__}: {err}")
        exit(0)


if __name__ == "__main__":
    main()
