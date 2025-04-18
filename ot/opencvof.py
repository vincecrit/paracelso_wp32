from ot.helpmsg import (FLAGS, FLOW, ITERATIONS, LEVELS, POLY_N, POLY_SIGMA,
                        PYR_SCALE, WINSIZE)

from .interfaces import BaseCLI


class OPENCVOF_CLI(BaseCLI):
    def __init__(self):
        super().__init__()
        self.add_specific_args()

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
