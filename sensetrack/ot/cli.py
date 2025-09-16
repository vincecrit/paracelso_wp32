"""
# CLI interface

This module provides a command-line interface (CLI) for the Offset-Tracking (OT) component of the sensetrack package.
It allows users to perform offset-tracking between two input images (reference and target) with configurable preprocessing
and output options.

## Classes

    BaseCLI: Base class for building CLI applications for offset-tracking algorithms. Handles argument parsing, image loading,
             preprocessing, algorithm execution, and output writing.

## Arguments

    -r, --reference          Path to the reference image (required).
    -t, --target             Path to the target image (required).
    -o, --output             Path to the output file (default: "output.tif").
    -b, --band               Band index to process (optional).
    --nodata                 Value to use for nodata pixels (optional).
    -prep, --preprocessing   Preprocessing method to apply (default: "equalize").
    --resultant_displacement Whether or not to export resultant displacement.
    If `False` exports two images for x and y displacements. Ignored for SKIPCCV output. Defaluts to `True`

### Typical workflow

    1. Parse command-line arguments.
    2. Load reference and target images.
    3. Apply preprocessing.
    4. Run the selected offset-tracking algorithm.
    5. Export the displacement results to the specified output file.
"""

import argparse
from pathlib import Path

from sensetrack.log import setup_logger
from sensetrack.ot import lib
from sensetrack.ot.helpmsg import BAND, NODATA, OUTPUT, REFERENCE, RES, TARGET
from sensetrack.ot.image_processing import dispatcher
from sensetrack.ot.interfaces import OTAlgorithm

logger = setup_logger(__name__)


class BaseCLI:
    def __init__(self, method: OTAlgorithm | None = None):
        self.parser = argparse.ArgumentParser(
            description="Offset-Tracking PARACELSO WP3.2"
        )
        self.add_common_args()

    def add_common_args(self):
        self.parser.add_argument(
            "-r", "--reference", required=True, help=REFERENCE, type=str
        )
        self.parser.add_argument("-t", "--target", required=True, help=TARGET, type=str)
        self.parser.add_argument(
            "-o", "--output", help=OUTPUT, default="output.tif", type=str
        )
        self.parser.add_argument("-b", "--band", help=BAND, default=None, type=int)
        self.parser.add_argument("--nodata", help=NODATA, default=None, type=float)
        self.parser.add_argument(
            "--resultant_displacement", help=RES, action="store_true"
        )
        self.parser.add_argument(
            "-prep", "--preprocessing", default="equalize", type=str
        )

    def get_parser(self):
        return self.parser

    def get_algorithm(self) -> OTAlgorithm: ...

    def run(self):

        parser = self.get_parser()
        args = parser.parse_args()
        algorithm = self.get_algorithm(**vars(args))

        reference, target = lib.load_images(
            args.reference, args.target, band=args.band, nodata=args.nodata
        )

        preprocessed_images = [
            dispatcher.dispatcher.dispatch_process(
                f"{algorithm.library}_{args.preprocessing}", array=img
            )
            for img in (reference, target)
        ]

        logger.info(f"{args.preprocessing.upper()} successfully completed.")
        output = algorithm(*preprocessed_images)
        logger.info(f"Algorithm {algorithm.__class__.__name__} successfully completed")

        if isinstance(output, dict):
            if args.resultant_displacement:  # export resultant displacements
                logger.info(f"Export to file: {args.output}")
                lib.write_output(output["res"], args.output)

            else:  # export displacement components (dxx, dyy)
                parent = Path(args.output).parent
                suffix = Path(args.output).suffix
                stem = Path(args.output).stem

                for cm in ["dxx", "dyy"]:
                    _outfile = parent / (stem + f"_{cm}{suffix}")
                    logger.info(f"Export {cm} to file: {_outfile}")
                    lib.write_output(output[cm], _outfile)

        else:  # export vector (for SKIPCCV algorithm)
            lib.write_output(output, args.output)
