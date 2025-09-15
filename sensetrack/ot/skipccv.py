"""
# CLI interface for `ot.algorithms.SkiPCC_Vector` algorithm

This module provides a command-line interface (CLI) for running the SkiPCC_Vector algorithm from the sensetrack.ot.algorithms package.
It extends the BaseCLI class to parse and handle specific arguments related to the SkiPCC_Vector algorithm.

# Classes

    SKIPCCV_CLI: Command-line interface class for configuring and running the `SkiPCC_Vector` algorithm.

# Arguments

    --phase_norm : bool (default: True)
        If set, applies phase normalization as described in the PHASENORM help message.
    --upsmp_fac : float (default: 1.0)
        Upsampling factor for the algorithm, as described in the UPSAMPLE_FACTOR help message.
    --step_size : int (default: 1)
        Step size for the algorithm, as described in the STEPSIZE help message.
    --winsize : int (default: 4)
        Window size for the algorithm, as described in the WINSIZE help message.
"""

from sensetrack.ot.algorithms import SkiPCC_Vector
from sensetrack.ot.cli import BaseCLI
from sensetrack.ot.helpmsg import PHASENORM, STEPSIZE, UPSAMPLE_FACTOR, WINSIZE


class SKIPCCV_CLI(BaseCLI):
    def __init__(self):
        super().__init__()
        self.add_specific_args()

    def get_algorithm(self, **kwargs):
        return SkiPCC_Vector.from_dict(kwargs)

    def add_specific_args(self):
        self.parser.add_argument(
            "--phase_norm", help=PHASENORM, action="store_true", default=True
        )
        self.parser.add_argument(
            "--upsmp_fac", help=UPSAMPLE_FACTOR, type=float, default=1.0
        )
        self.parser.add_argument("--step_size", help=STEPSIZE, type=int, default=1)
        self.parser.add_argument("--winsize", help=WINSIZE, type=int, default=4)


if __name__ == "__main__":
    cli = SKIPCCV_CLI()
    cli.run()
