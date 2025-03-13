import logging

def fetch_logger():
    FORMATSTR = "%(message)s"
    FORMATSTR_DEBUG = "%(asctime)s [%(name)s.%(funcName)s - " +\
        "%(lineno)s][%(levelname)s] - %(message)s"
    DATEFMT = "%Y.%m.%d-%H:%M:%S"

    logger = logging.getLogger(__name__)

    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    fh = logging.FileHandler(filename=__name__+'.log')

    ch.setLevel(logging.INFO)
    fh.setLevel(logging.DEBUG)

    ch.setFormatter(logging.Formatter(FORMATSTR))
    fh.setFormatter(logging.Formatter(FORMATSTR_DEBUG, datefmt=DATEFMT))

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.propagate = True
    
    return logger