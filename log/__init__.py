import logging
import os
from logging.handlers import TimedRotatingFileHandler

LOG_FORMAT = "%(asctime)s [%(name)s.%(funcName)s - %(lineno)s][%(levelname)s] - %(message)s"
LOG_LEVEL = logging.INFO  # Replace 200 with a valid logging level


def setup_logger(name: str):
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
    file_handler = TimedRotatingFileHandler(filename="log.txt", when='D', interval=1)
    file_handler.suffix = "%Y%m%d"
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)
    
    return logger
