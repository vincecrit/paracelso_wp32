# sensetrack.prs - Post-Processing and Conversion Submodule

## Overview
Il sottopacchetto `sensetrack.prs` fornisce strumenti per la post-elaborazione e la conversione dei risultati di tracking e dati PRISMA, facilitando l'integrazione con software GIS e pipeline di analisi.

## Struttura del modulo
- `convert.py`
  - Funzioni per:
    - Lettura di file HDF5 PRISMA.
    - Estrazione di metadati e informazioni di prodotto (`get_prisma_info`).
    - Conversione di bande PRISMA (pan, swir, vnir) in GeoTIFF georeferenziati (`prisma_panchromatic_to_gtiff`).
    - Parsing degli argomenti da linea di comando per conversione e info.

## Funzionalità principali
- **Estrazione metadati PRISMA**: stampa informazioni chiave (ID prodotto, livello di processing, percentuale nuvolosità, EPSG) da file HDF5 PRISMA.
- **Conversione a GeoTIFF**: esporta bande panchromatiche, SWIR e VNIR da HDF5 PRISMA a file GeoTIFF georeferenziati, mantenendo proiezione e bounding box.
- **CLI**: permette di lanciare conversioni e info direttamente da terminale con opzioni dedicate.

## Esempio di utilizzo
### Da Python
```python
from sensetrack.prs.convert import get_prisma_info, prisma_panchromatic_to_gtiff

# Estrai info da un file PRISMA
get_prisma_info('path/to/file.h5')

# Converte la banda SWIR in GeoTIFF
prisma_panchromatic_to_gtiff('path/to/file.h5', band='swir')
```

### Da terminale
```powershell
python sensetrack/prs/convert.py -f path/to/file.h5 --convert
python sensetrack/prs/convert.py -f path/to/file.h5 --show_info
```

## Note
- I blocchi di bande supportate per la conversione sono: 'pan', 'swir', 'vnir'.
- L'output GeoTIFF mantiene la georeferenziazione e la proiezione EPSG del prodotto PRISMA.
- Il modulo può essere esteso per supportare altri formati o funzioni di post-processing.

Per dettagli sui parametri e sulle opzioni disponibili, consultare la documentazione inline nei file sorgente.
