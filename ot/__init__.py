import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

FORMATSTR = "%(asctime)s [%(name)s.%(funcName)s - "+\
    "%(lineno)s][%(levelname)s] - %(message)s"
    
DATEFMT = "%Y.%M.%d-%H:%M:%S"

formatter = logging.Formatter(FORMATSTR, datefmt=DATEFMT)

ch.setFormatter(formatter)

logger.addHandler(ch)