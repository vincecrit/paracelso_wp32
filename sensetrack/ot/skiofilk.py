from sensetrack.ot.algoritmi import SkiOpticalFlowILK
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
