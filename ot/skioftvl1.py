from ot.helpmsg import ATTACHMENT, NUMITER, NUMWARP, PREFILTER, TIGHTNESS, TOL

from .interfaces import BaseCLI


class SKIOFTVL1_CLI(BaseCLI):
    def __init__(self):
        super().__init__()
        self.add_specific_args()

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
