from abc import ABC, abstractmethod
import cv2
import numpy as np
from scipy.ndimage import correlate
from skimage.feature import match_template

# Interfaccia comune per tutti gli algoritmi di offset tracking, con parametri aggiuntivi
class OffsetTrackingStrategy(ABC):
    
    @abstractmethod
    def compute_offset(self, image1, image2, params: dict):
        pass


# Implementazione con OpenCV
class OpenCVPixelOffset(OffsetTrackingStrategy):
    
    def compute_offset(self, image1, image2, params: dict):
        # Recupera parametri specifici per OpenCV
        precision = params.get("precision", 0.001)  # valore di default
        
        # Algoritmo base con OpenCV che usa il parametro precision
        result = cv2.phaseCorrelate(np.float32(image1), np.float32(image2), precision)
        
        return {"offset": result, "precision": precision}


# Implementazione con Scipy
class ScipyPixelOffset(OffsetTrackingStrategy):
    
    def compute_offset(self, image1, image2, params: dict):
        # Recupera un eventuale parametro di metodo per Scipy
        mode = params.get("mode", "constant")  # valore di default
        
        # Algoritmo base con Scipy che usa il parametro mode
        result = correlate(image1, image2, mode=mode)
        
        return {"correlation": result, "mode": mode}


# Implementazione con scikit-image [DA CONTROLLARE ASSOLUTAMENTE CHE NON MI FIDO]
class SkimageOffsetTracking(OffsetTrackingStrategy):
    
    def compute_offset(self, image1, image2, params: dict):
        # Recupera eventuali parametri per scikit-image
        pad_input = params.get("pad_input", True)  # valore di default
        
        # Algoritmo base con scikit-image che usa il parametro pad_input
        result = match_template(image1, image2, pad_input=pad_input)
        
        return {"match": result, "pad_input": pad_input}


# Factory per la creazione degli algoritmi
class OffsetTrackingFactory:
    
    @staticmethod
    def get_algorithm(algorithm_type: str) -> OffsetTrackingStrategy:
        if algorithm_type == 'opencv':
            return OpenCVPixelOffset()
        elif algorithm_type == 'scipy':
            return ScipyPixelOffset()
        elif algorithm_type == 'skimage':
            return SkimageOffsetTracking()
        else:
            raise ValueError(f"Algoritmo '{algorithm_type}' non supportato.")


# Factory per la creazione degli algoritmi
class PixelTrackingFactory:
    
    @staticmethod
    def get_algorithm(algorithm_type: str) -> OffsetTrackingStrategy:
        raise NotImplementedError


class PointTrackingFactory:
    
    @staticmethod
    def get_algorithm(algorithm_type: str) -> OffsetTrackingStrategy:
        if algorithm_type == 'scipy':
            return ScipyPixelOffset()
        elif algorithm_type == 'skimage':
            return SkimageOffsetTracking()
        else:
            raise ValueError(f"Algoritmo '{algorithm_type}' non supportato.")


class QuiverTrackingFactory:
    
    @staticmethod
    def get_algorithm(algorithm_type: str) -> OffsetTrackingStrategy:
        if algorithm_type == 'opencv':
            return NotImplementedError
        elif algorithm_type == 'scipy':
            return NotImplementedError
        elif algorithm_type == 'skimage':
            return NotImplementedError
        else:
            raise ValueError(f"Algoritmo '{algorithm_type}' non supportato.")


# Servizio che utilizza la factory e esegue il tracking con parametri aggiuntivi
class OffsetTrackingService:
    
    def __init__(self, algorithm_type: str):
        self.algorithm = OffsetTrackingFactory.get_algorithm(algorithm_type)
    
    def process_images(self, image1, image2, params: dict):
        return self.algorithm.compute_offset(image1, image2, params)


# Esempio d'uso
if __name__ == "__main__":
    # Immagini di input (dovrebbero essere caricate da file o da un'API)
    image1 = np.random.rand(100, 100)
    image2 = np.random.rand(100, 100)
    
    # Seleziona l'algoritmo da usare (dalla web app o configurazione)
    algorithm_type = 'opencv'  # 'scipy', 'skimage'
    
    # Parametri aggiuntivi (recuperati dall'utente o dall'interfaccia web)
    params = {
        "precision": 0.0001,  # solo per opencv
        "mode": "reflect",    # solo per scipy
        "pad_input": False    # solo per scikit-image
    }
    
    # Crea il servizio
    tracking_service = PixelTrackingFactory(algorithm_type)
    
    # Esegui il processamento
    result = tracking_service.process_images(image1, image2, params)
    
    print(f"Risultato con algoritmo {algorithm_type}: {result}")
