import argparse
from pathlib import Path

import numpy as np
import rasterio

from ot.coreg import basic_pixel_coregistration
from ot.helpmsg import (ALGNAME, ATTACHMENT, BAND, FLAGS, FLOW, GAUSSIAN,
                        ITERATIONS, LEVELS, LOGNORM, NODATA, NORMALIZE,
                        NUMITER, NUMWARP, OUTPUT, POLY_N, POLY_SIGMA,
                        POWERNORM, PREFILTER, PYR_SCALE, RADIUS, REFERENCE,
                        RGB2GRAY, TARGET, TIGHTNESS, TOL, WINSIZE)
from ot.metodi import get_algorithm
from ot.normalize import lognorm, normalize, powernorm, rgb2single_band


def get_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description="Optical flow")
    # parametri comuni
    parser.add_argument("-ot", "--algname", help=ALGNAME, type=str)
    parser.add_argument("-r", "--reference", help=REFERENCE, type=str)
    parser.add_argument("-t", "--target", help=TARGET, type=str)
    parser.add_argument("-o", "--output", help=OUTPUT,
                        default="output.tif", type=str)
    parser.add_argument("-b", "--band", help=BAND, default=None, type=int)
    parser.add_argument("--nodata", help=NODATA, default=None, type=float)
    parser.add_argument("--rgb2gray", help=RGB2GRAY, action="store_true")
    parser.add_argument("--lognorm", help=LOGNORM,
                        default=False, action="store_true")
    parser.add_argument("--normalize", help=NORMALIZE,
                        default=False, action="store_true")
    parser.add_argument("--powernorm", help=POWERNORM,
                        default=None, type=float)
    # parametri OpenCV
    parser.add_argument("--flow", help=FLOW, default=False)
    parser.add_argument("--levels", help=LEVELS, default=4, type=int)
    parser.add_argument("--pyr_scale", help=PYR_SCALE, default=0.5, type=float)
    parser.add_argument("--winsize", help=WINSIZE, type=int, default=4)
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

    return parser


def main() -> None:
    args = get_parser().parse_args()
    algorithm = get_algorithm(args.algname)
    OTMethod = algorithm.from_dict(vars(args))

    target_coreg = Path(args.target).parent / \
        (Path(args.target).stem+"_coreg.tif")

    basic_pixel_coregistration(args.target, args.reference, target_coreg)

    with rasterio.open(args.reference) as refimg:
        with rasterio.open(target_coreg) as tarimg:

            reference_array = refimg.read(args.band).astype(np.float32)
            target_array = tarimg.read(args.band).astype(np.float32)

            if args.nodata is not None:
                NODATA3B = NODATA1B = args.nodata
            else:
                NODATA3B, NODATA1B = -9999, reference_array.min()

            if reference_array.ndim > 2:
                target_mask = reference_array[-1, :, :] == NODATA3B
                reference_array[0, :, :][target_mask] = NODATA3B
                reference_array[1, :, :][target_mask] = NODATA3B

            else:
                target_mask = reference_array == NODATA1B
                reference_array = np.ma.masked_array(
                    reference_array, target_mask)
                target_array = np.ma.masked_array(target_array, target_mask)

            target_metadata = tarimg.meta.copy()
            target_metadata["nodata"] = NODATA1B

    if args.rgb2gray:
        if args.band is not None:
            print("ERRORE. Se si vuole convertire in scala di grigi, non è possibile specificare una singola banda")
            exit(1)

        elif all([args.rgb2gray, args.band is not None]) or all([not args.rgb2gray, args.band is None]):
            print("ERRORE. Specificare uno tra rgb2gray e band")
            exit(1)

        reference_array = np.ma.masked_array(
            rgb2single_band(reference_array), target_mask)

        target_array = np.ma.masked_array(
            rgb2single_band(target_array), target_mask)

    # if args.norm:
    if args.lognorm:
        if np.any(reference_array <= 0) or np.any(target_array <= 0):
            print("ERRORE. LogNorm non può essere applicata a valori negativi")
            exit(1)

        reference_array = lognorm(reference_array)
        target_array = lognorm(target_array)

    elif args.powernorm:  # se lognorm e powernorm sono entrambi True, prevale lognorm
        reference_array = powernorm(reference_array, args.powernorm)
        target_array = powernorm(target_array, args.powernorm)

    else:  # se powernorm è True, prevale powernorm
        reference_array = normalize(reference_array)
        target_array = normalize(target_array)

    target_metadata["count"] = 1

    resdispl = OTMethod(target_metadata, reference_array, target_array)

    target_metadata["dtype"] = "float32"
    resdispl_ma = np.ma.masked_array(resdispl, target_mask)
    print(np.min(resdispl_ma))
    print(np.max(resdispl_ma))

    with rasterio.open(args.output, "w", **target_metadata) as dst:
        dst.write(resdispl, 1)


if __name__ == "__main__":
    main()
