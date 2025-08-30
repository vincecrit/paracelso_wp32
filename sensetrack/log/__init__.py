"""
sensetrack.log
--------------
This module provides a utility function to configure and retrieve a logger instance
with a predefined format and log level for the sensetrack package. The logger outputs
to the console and can be extended to support file logging with daily rotation.
Functions:
    setup_logger(name: str): Configures and returns a logger with the specified name,
    using a consistent format and INFO log level. Prevents duplicate handlers and
    disables propagation to ancestor loggers.
"""

import logging

LOG_FORMAT = "%(asctime)s [%(name)s.%(funcName)s - %(lineno)s][%(levelname)s] - %(message)s"
LOG_LEVEL = logging.DEBUG


def setup_logger(name: str, LOG_LEVEL: int = LOG_LEVEL):
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    logger.propagate = False  # Evita la propagazione ai logger superiori

    # Rimuove eventuali handler duplicati
    if logger.hasHandlers():
        logger.handlers.clear()

    # Handler per la console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)

    # Handler per il file (a tempo, un file ogni giorno)
    # file_handler = TimedRotatingFileHandler(filename="log.txt", when='D', interval=1)
    # file_handler.suffix = "%Y%m%d"
    # file_handler.setLevel(LOG_LEVEL)
    # file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    # logger.addHandler(file_handler)

    return logger
