from ot.helpmsg import PHASENORM, STEPSIZE, UPSAMPLE_FACTOR, WINSIZE

from .interfaces import BaseCLI


class SKIPCCV_CLI(BaseCLI):
    def __init__(self):
        super().__init__()
        self.add_specific_args()

    def add_specific_args(self):
        self.parser.add_argument("--phase_norm", help=PHASENORM,
                                 action="store_true", default=True)
        self.parser.add_argument("--upsmp_fac", help=UPSAMPLE_FACTOR,
                                 type=float, default=1.0)
        self.parser.add_argument(
            "--step_size", help=STEPSIZE, type=int, default=1)
        self.parser.add_argument(
            "--winsize", help=WINSIZE, type=int, default=4)


if __name__ == "__main__":
    cli = SKIPCCV_CLI()
    cli.run()
