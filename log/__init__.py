import logging
import os

# Imposta il formato del log
LOG_FORMAT = "%(asctime)s [%(name)s.%(funcName)s - %(lineno)s][%(levelname)s] - %(message)s"
LOG_LEVEL = logging.DEBUG  # Puoi cambiarlo a INFO, WARNING, ERROR, CRITICAL
LOG_FILE = "wp32.log"

# Crea una cartella per i log se non esiste
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, LOG_FILE)

# Configura il logger principale
def setup_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    # Evita duplicati di handler
    if not logger.handlers:
        # Handler per la console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LOG_LEVEL)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        
        # Handler per il file
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(LOG_LEVEL)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        
        # Aggiunge gli handler al logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger

# Esempio di utilizzo nei vari moduli
def main():
    logger = setup_logger("main")
    logger.info("Sistema di logging inizializzato con successo.")
    logger.debug("Debugging attivo per il progetto.")
    
if __name__ == "__main__":
    main()
