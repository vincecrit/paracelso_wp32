from log import setup_logger
from ot.image_processing import opencv, ski
from ot.interfaces import PreprocessDispatcher

logger = setup_logger(__name__)

dispatcher = PreprocessDispatcher()

for funcname in ski.__all__:
    try:
        dispatcher.register("skimage_"+funcname, eval(f"ski.{funcname}"))
    except AttributeError as err:
        logger.critical(f"{funcname} non è tra i metodi registrabili")
        exit(0)

for funcname in opencv.__all__:
    try:
        dispatcher.register("OpenCV_"+funcname, eval(f"opencv.{funcname}"))
    except AttributeError as err:
        logger.critical(f"{funcname} non è tra i metodi registrabili")
        exit(0)