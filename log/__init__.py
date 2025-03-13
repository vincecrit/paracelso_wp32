import logging
import os
from logging.handlers import TimedRotatingFileHandler

LOG_FORMAT = "%(asctime)s [%(name)s.%(funcName)s - %(lineno)s][%(levelname)s] - %(message)s"
LOG_LEVEL = logging.DEBUG  # Puoi cambiarlo a INFO, WARNING, ERROR, CRITICAL

LOG_FILE = "wp32.log"
LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, LOG_FILE)

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
    file_handler = TimedRotatingFileHandler(log_path, when='d', interval=1)
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)
    
    return logger
