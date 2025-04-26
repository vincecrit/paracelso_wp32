import argparse
from pathlib import Path

from log import setup_logger
from ot import lib
from ot.helpmsg import BAND, NODATA, OUTPUT, REFERENCE, TARGET
from ot.image_processing import dispatcher
from ot.interfaces import OTAlgorithm

logger = setup_logger(__name__)


class BaseCLI:
    def __init__(self, method: OTAlgorithm | None = None):
        self.parser = argparse.ArgumentParser(
            description="Offset-Tracking PARACELSO WP3.2")
        self.add_common_args()

    def add_common_args(self):
        self.parser.add_argument(
            "-r", "--reference", required=True, help=REFERENCE, type=str)
        self.parser.add_argument(
            "-t", "--target", required=True, help=TARGET, type=str)
        self.parser.add_argument(
            "-o", "--output", help=OUTPUT, default="output.tif", type=str)
        self.parser.add_argument(
            "-b", "--band", help=BAND, default=None, type=int)
        self.parser.add_argument(
            "--nodata", help=NODATA, default=None, type=float)
        self.parser.add_argument(
            "-prep", "--preprocessing", default='equalize', type=str)
        # self.parser.add_argument(
        #     "-ot", "--algname", required=True, help=ALGNAME, type=str)
        # self.parser.add_argument(
        #     "--out_format", default=None, type=str)

    def get_parser(self): return self.parser

    def get_algorithm(self): return None

    def run(self):

        parser = self.get_parser()
        args = parser.parse_args()
        algorithm = self.get_algorithm(**vars(args))

        reference, target = lib.load_images(
            args.reference, args.target, band=args.band, nodata=args.nodata)

        preprocessed_images = [
            dispatcher.dispatcher.dispatch_process(
                f"{algorithm.library}_{args.preprocessing}", array=img)
            for img in (reference, target)]

        logger.info(f"{args.preprocessing.upper()} eseguito correttamente.")
        displacements = algorithm(*preprocessed_images)
        logger.info(
            f"Algoritmo {algorithm.__class__.__name__} eseguito correttamente")

        logger.info(f"Esporto su file: {args.output}")
        lib.write_output(displacements, args.output)
