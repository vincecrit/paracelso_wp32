import argparse
from pathlib import Path

import geopandas as gpd

from ot.coreg import basic_pixel_coregistration
from ot.helpmsg import (ALGNAME, ATTACHMENT, BAND, FLAGS, FLOW, GAUSSIAN,
                        ITERATIONS, LEVELS, LOGNORM, MINMAX, NODATA, NORMALIZE,
                        NUMITER, NUMWARP, OUTPUT, POLY_N, POLY_SIGMA, CLAHE,
                        PREFILTER, PYR_SCALE, RADIUS, REFERENCE, TARGET, UPSAMPLE_FACTOR,
                        TIGHTNESS, TOL, WINSIZE, ZSCORENORM, STEPSIZE, PHASENORM)
from ot.interfaces import Image, OTAlgorithm
from ot.metodi import get_algorithm


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
    parser.add_argument("--lognorm", help=LOGNORM,
                        default=False, action="store_true")
    parser.add_argument("--normalize", help=NORMALIZE,
                        default=False, action="store_true")
    parser.add_argument("--zscore", help=ZSCORENORM,
                        default=None, action="store_true")
    parser.add_argument("--minmax", help=MINMAX,
                        default=None, action="store_true")
    parser.add_argument("--clahe", help=CLAHE,
                        default=None, action="store_true")
    # parametri OpenCV
    parser.add_argument("--flow", help=FLOW, default=False)
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
    """Main function to execute the optical flow algorithm."""
    args = get_parser().parse_args()
    algorithm = get_algorithm(args.algname)
    OTMethod: OTAlgorithm = algorithm.from_dict(vars(args))

    reference = Image.from_file(args.reference, nodata=args.nodata)
    target = Image.from_file(args.target, nodata=args.nodata)

    if not reference.is_coregistered(target):
        target_coreg = Path(args.target).parent / \
            (Path(args.target).stem+"_coreg.tif")

        basic_pixel_coregistration(args.target, args.reference, target_coreg)
        target = Image.from_file(target_coreg, nodata=args.nodata)

    if args.lognorm:
        reference = reference.log_norm()
        target = target.log_norm()

    elif args.zscore:  # se lognorm e zscore sono entrambi True, prevale lognorm
        reference = reference.zscore_norm()
        target = target.zscore_norm()

    elif args.minmax:  # se zscore è True, prevale zscore
        reference = reference.minmax_norm()
        target = target.minmax_norm()

    elif args.clahe:  # se zscore è True, prevale zscore
        reference = reference.cla()
        target = target.minmax_norm()

    if args.band is not None:
        _reference = reference.get_band(args.band)
        _target = target.get_band(args.band)
    
    else:
        _reference = reference.to_single_band()
        _target = target.to_single_band()

    # OFFSET TRACKING
    print(_reference.image.shape)
    print(_target.image.shape)
    result = OTMethod(_reference, _target)

    if isinstance(result, (Image, gpd.GeoDataFrame)):
        result.to_file(args.output)


if __name__ == "__main__":
    main()
