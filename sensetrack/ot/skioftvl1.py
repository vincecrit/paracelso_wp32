"""
skioftvl1.py
This module provides a command-line interface (CLI) for the SkiOpticalFlowTVL1 algorithm, 
which estimates optical flow using the TV-L1 method. The CLI allows users to configure 
algorithm parameters via command-line arguments.

Classes:

    SKIOFTVL1_CLI: Inherits from BaseCLI and implements argument parsing and algorithm instantiation for `SkiOpticalFlowTVL1` algorithm.

Arguments:

    --num_warp (int, default=3): 
        Number of warping iterations to perform during the optical flow estimation.
    --prefilter (flag, default=False): 
        If set, applies a prefiltering step to the input data before processing.
    --attachment (int, default=10): 
        Data attachment weight parameter for the TV-L1 algorithm.
    --tightness (float, default=0.3): 
        Tightness parameter (lambda) controlling the smoothness of the estimated flow.
    --num_iter (int, default=10): 
        Number of iterations for the optimization process.
    --tol (float, default=1e-4): 
        Tolerance for convergence; the algorithm stops if the change is below this threshold.
"""

from sensetrack.ot.algorithms import SkiOpticalFlowTVL1
from sensetrack.ot.cli import BaseCLI
from sensetrack.ot.helpmsg import (ATTACHMENT, NUMITER, NUMWARP, PREFILTER,
                                   TIGHTNESS, TOL)


class SKIOFTVL1_CLI(BaseCLI):
    def __init__(self):
        super().__init__()
        self.add_specific_args()

    def get_algorithm(self, **kwargs):
        return SkiOpticalFlowTVL1.from_dict(kwargs)

    def add_specific_args(self):
        self.parser.add_argument(
            "--num_warp", help=NUMWARP, type=int, default=3)
        self.parser.add_argument(
            "--prefilter", help=PREFILTER, action="store_true")
        self.parser.add_argument(
            "--attachment", help=ATTACHMENT, type=int, default=10)
        self.parser.add_argument(
            "--tightness", help=TIGHTNESS, type=float, default=0.3)
        self.parser.add_argument(
            "--num_iter", help=NUMITER, type=int, default=10)
        self.parser.add_argument(
            "--tol", help=TOL, type=float, default=1e-4)


if __name__ == "__main__":
    cli = SKIOFTVL1_CLI()
    cli.run()
