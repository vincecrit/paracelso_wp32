# import logging

# FORMATSTR = "%(message)s"
# FORMATSTR_DEBUG = "%(asctime)s [%(name)s.%(funcName)s - %(lineno)s][%(levelname)s] - %(message)s"
# DATEFMT = "%Y.%m.%d-%H:%M:%S"

# logger = logging.getLogger(__name__)

# logger.setLevel(logging.DEBUG)

# if not logger.handlers:
#     ch = logging.StreamHandler()
#     fh = logging.FileHandler(filename=__name__+'.log')

#     ch.setLevel(logging.DEBUG)
#     fh.setLevel(logging.DEBUG)

#     ch.setFormatter(logging.Formatter(FORMATSTR_DEBUG))
#     fh.setFormatter(logging.Formatter(FORMATSTR_DEBUG, datefmt=DATEFMT))

#     logger.addHandler(ch)
#     logger.addHandler(fh)
