"""
opencvof.py
This module provides a command-line interface (CLI) for configuring and running the OpenCV-based optical flow algorithm.
It defines the `OPENCVOF_CLI` class, which extends the generic `BaseCLI` to add specific arguments relevant to optical flow computation,
such as flow type, pyramid scale, window size, number of iterations, polynomial expansion parameters, and algorithm flags.

Classes:

    OPENCVOF_CLI: Inherits from BaseCLI and implements argument parsing and algorithm instantiation for `OpenCVOpticalFl

Arguments:

    --flow         : Type of optical flow algorithm to use.
    --levels       : Number of pyramid layers including the initial image.
    --pyr_scale    : Image scale (<1) to build pyramids for each image.
    --winsize      : Window size dimension.
    --iterations   : Number of iterations at each pyramid level.
    --poly_n       : Size of the pixel neighborhood used to find polynomial expansion.
    --poly_sigma   : Standard deviation of the Gaussian used to smooth derivatives.
    --flags        : Operation flags for the algorithm.

Dependencies:

    - ot.helpmsg: Contains help messages for CLI arguments.
    - sensetrack.ot.algoritmi: Provides the OpenCVOpticalFlow algorithm implementation.
    - sensetrack.ot.cli: BaseCLI class for command-line interfaces.
"""
from sensetrack.ot.helpmsg import (FLAGS, FLOW, ITERATIONS, LEVELS, POLY_N, POLY_SIGMA,
                        PYR_SCALE, WINSIZE)

from sensetrack.ot.algorithms import OpenCVOpticalFlow
from sensetrack.ot.cli import BaseCLI


class OPENCVOF_CLI(BaseCLI):
    def __init__(self):
        super().__init__()
        self.add_specific_args()

    def get_algorithm(self, **kwargs):
        return OpenCVOpticalFlow.from_dict(kwargs)

    def add_specific_args(self):
        self.parser.add_argument(
            "--flow", help=FLOW, default=None)
        self.parser.add_argument(
            "--levels", help=LEVELS, default=4, type=int)
        self.parser.add_argument(
            "--pyr_scale", help=PYR_SCALE, default=0.5, type=float)
        self.parser.add_argument(
            "--winsize", help=WINSIZE, type=int, default=4)
        self.parser.add_argument(
            "--iterations", help=ITERATIONS, default=10, type=int)
        self.parser.add_argument(
            "--poly_n", help=POLY_N, default=5, type=int)
        self.parser.add_argument(
            "--poly_sigma", help=POLY_SIGMA, default=1.1, type=float)
        self.parser.add_argument(
            "--flags", help=FLAGS, default=None, type=int)


if __name__ == "__main__":
    cli = OPENCVOF_CLI()
    cli.run()
