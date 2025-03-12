import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
fh = logging.FileHandler(filename=__name__+'.log')

ch.setLevel(logging.DEBUG)
fh.setLevel(logging.DEBUG)

FORMATSTR = "%(asctime)s [%(name)s.%(funcName)s - "+\
    "%(lineno)s][%(levelname)s] - %(message)s"
    
DATEFMT = "%Y.%m.%d-%H:%M:%S"

formatter = logging.Formatter(FORMATSTR, datefmt=DATEFMT)

ch.setFormatter(formatter)
fh.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)