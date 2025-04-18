from log import setup_logger
from ot.image_processing import opencv, ski

logger = setup_logger(__name__)


class PreprocessDispatcher:
    def __init__(self):
        self.processes = dict()

    def register(self, name: str, process):
        if name not in self.processes:
            self.processes[name] = list()
        self.processes[name].append(process)

    def dispatch_process(self, name: str, **kwargs):
        if not name in self.processes:
            logger.critical(
                f"Il metodo {name.upper()} non è tra quelli registrati: " +
                f"{self.processes.keys()}")
            exit(0)
        else:
            for process in self.processes[name]:
                return process(**kwargs)


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
