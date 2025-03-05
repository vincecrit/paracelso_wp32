import logging

from ot.interfaces import PreprocessDispatcher

from . import opencv, ski

logger = logging.getLogger(__name__)

# registrazione metodi
dispatcher = PreprocessDispatcher()

for funcname in ski.__all__:
    try:
        dispatcher.register("ski_"+funcname, eval(f"ski.{funcname}"))
    except AttributeError as err:
        logger.critical(f"{funcname} non è tra i metodi registrabili")
        exit(0)

for funcname in opencv.__all__:
    try:
        dispatcher.register("cv2_"+funcname, eval(f"opencv.{funcname}"))
    except AttributeError as err:
        logger.critical(f"{funcname} non è tra i metodi registrabili")
        exit(0)