"""
This module implements a dispatcher pattern for image preprocessing operations,
allowing registration and execution of both OpenCV and scikit-image based processing methods.
"""

from sensetrack.log import setup_logger
from sensetrack.ot.image_processing import opencv, ski

logger = setup_logger(__name__)


class PreprocessDispatcher:
    """
    A dispatcher class that manages and executes image preprocessing operations.

    This class implements a registry pattern where different preprocessing methods
    can be registered and then dispatched by name. It supports both OpenCV and
    scikit-image based processing methods.
    """

    def __init__(self):
        """
        Initialize the PreprocessDispatcher with an empty process registry.
        """
        self.processes = dict()

    def register(self, name: str, process):
        """
        Register a new preprocessing method under a given name.

        Args:
            name (str): The name under which to register the process
            process: The processing function to register
        """
        if name not in self.processes:
            self.processes[name] = list()
        self.processes[name].append(process)

    def dispatch_process(self, name: str, **kwargs):
        """
        Execute a registered preprocessing method by its name.

        Args:
            name (str): The name of the process to execute
            **kwargs: Additional keyword arguments to pass to the process

        Returns:
            The result of the executed process

        Exits:
            If the requested process name is not found in the registry

        # Examples:

        ## Retrieving the CLAHE function for OpenCV-based optical flow
        >>> from sensetrack.ot.algorithms import OpenCVOpticalFlow
        >>> from senstrack.ot.image_processing import dispatcher
        >>> from sensetrack.ot import lib
        ### Loading images (they will be coregistered if needed)
        >>> ref, tar = lib.load_images("ref.tif", "tar.tif")
        ### OpenCV optical flow
        >>> OT = OpenCVOpticalFlow.from_dict({"pyr_scale":0.5, "levels":4, "winsize":16})
        ### OT instance has its `library` attribute:
        >>> print(OT.library)
        'OpenCV'
        ### The dispatcher can be called:
        >>> ref_clahe = dispatcher.dispatch_process(f"{OT.library}_clahe", array = ref)
        >>> tar_clahe = dispatcher.dispatch_process(f"{OT.library}_clahe", array = tar)
        """
        if not name in self.processes:
            logger.critical(
                f"Method {name.upper()} is not among registered ones: "
                + f"{self.processes.keys()}"
            )
            exit(0)
        else:
            for process in self.processes[name]:
                return process(**kwargs)


# Create the global dispatcher instance
dispatcher = PreprocessDispatcher()

# Register scikit-image methods
for funcname in ski.__all__:
    try:
        dispatcher.register("skimage_" + funcname, eval(f"ski.{funcname}"))
    except AttributeError as err:
        logger.critical(f"{funcname} is not among registerable methods")
        exit(0)

# Register OpenCV methods
for funcname in opencv.__all__:
    try:
        dispatcher.register("OpenCV_" + funcname, eval(f"opencv.{funcname}"))
    except AttributeError as err:
        logger.critical(f"{funcname} is not among registerable methods")
        exit(0)
