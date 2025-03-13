import logging
import os
<<<<<<< HEAD
from logging.handlers import TimedRotatingFileHandler

LOG_FORMAT = "%(asctime)s [%(name)s.%(funcName)s - %(lineno)s][%(levelname)s] - %(message)s"
LOG_LEVEL = logging.DEBUG  # Puoi cambiarlo a INFO, WARNING, ERROR, CRITICAL

LOG_FILE = "wp32.log"
LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, LOG_FILE)

=======

# Imposta il formato del log
LOG_FORMAT = "%(asctime)s [%(name)s.%(funcName)s - %(lineno)s][%(levelname)s] - %(message)s"
LOG_LEVEL = logging.DEBUG  # Puoi cambiarlo a INFO, WARNING, ERROR, CRITICAL
LOG_FILE = "wp32.log"

# Crea una cartella per i log se non esiste
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, LOG_FILE)

# Configura il logger principale
>>>>>>> 2187fe81376308a0fd0384d449fffc33cbcab761
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
    
<<<<<<< HEAD
    # Handler per il file (a tempo, un file ogni giorno)
    file_handler = TimedRotatingFileHandler(log_path, when='d', interval=1)
=======
    # Handler per il file
    file_handler = logging.FileHandler(log_path, mode='a')
>>>>>>> 2187fe81376308a0fd0384d449fffc33cbcab761
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)
    
    return logger
<<<<<<< HEAD
=======

# Esempio di utilizzo nei vari moduli
def main():
    logger = setup_logger("main")
    logger.info("Sistema di logging inizializzato con successo.")
    logger.debug("Debugging attivo per il progetto.")
    
if __name__ == "__main__":
    main()
>>>>>>> 2187fe81376308a0fd0384d449fffc33cbcab761
