"""
skiofilk.py
This module provides a command-line interface (CLI) for configuring and running the `SkiOpticalFlowILK` algorithm, 
which is part of the sensetrack optical flow toolkit. The CLI allows users to specify algorithm parameters such as 
radius, number of warps, Gaussian smoothing, and prefiltering via command-line arguments.

Classes:

    SKIOFILK_CLI: Inherits from BaseCLI and implements argument parsing and algorithm instantiation for `SkiOpticalFlowILK`.

Arguments:

    --radius (int): Neighborhood radius for the optical flow algorithm. Default is 4.
    --num_warp (int): Number of warping iterations. Default is 3.
    --gaussian: Enable Gaussian smoothing. Optional flag.
    --prefilter: Enable prefiltering. Optional flag.
"""
from sensetrack.ot.algorithms import SkiOpticalFlowILK
from sensetrack.ot.cli import BaseCLI
from sensetrack.ot.helpmsg import GAUSSIAN, NUMWARP, PREFILTER, RADIUS


class SKIOFILK_CLI(BaseCLI):
    def __init__(self):
        super().__init__()
        self.add_specific_args()

    def get_algorithm(self, **kwargs):
        return SkiOpticalFlowILK.from_dict(kwargs)

    def add_specific_args(self):
        self.parser.add_argument(
            "--radius", help=RADIUS, type=int, default=4)
        self.parser.add_argument(
            "--num_warp", help=NUMWARP, type=int, default=3)
        self.parser.add_argument(
            "--gaussian", help=GAUSSIAN, action="store_true")
        self.parser.add_argument(
            "--prefilter", help=PREFILTER, action="store_true")


if __name__ == "__main__":
    cli = SKIOFILK_CLI()
    cli.run()
