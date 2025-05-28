# sensetrack.ot - Offset-Tracking Submodule

## Overview
Il sottopacchetto `sensetrack.ot` fornisce le funzionalità principali per l'analisi di optical flow, la normalizzazione delle immagini, la gestione delle interfacce e la CLI per l'offset tracking. È progettato per lavorare con immagini satellitari e dati raster, offrendo algoritmi avanzati e strumenti di supporto per la ricerca e l'applicazione operativa.

## Struttura del modulo
- `algoritmi.py`  
  Implementa le classi e le funzioni per il calcolo dell'optical flow e della cross-correlazione di fase tra immagini. Fornisce wrapper per algoritmi OpenCV e scikit-image, oltre a utilità per la conversione dei risultati in DataFrame o GeoDataFrame.

- `cli.py`  
  Implementa la Command Line Interface per lanciare i processi di optical flow, normalizzazione e altre operazioni direttamente da terminale.

- `helpmsg.py`  
  Contiene i messaggi di aiuto e la documentazione testuale per la CLI e le funzioni principali del modulo.

- `interfaces.py`  
  Definisce le classi per la rappresentazione delle immagini, la gestione delle bande e le interfacce astratte per gli algoritmi di tracking.

- `lib.py`  
  Funzioni di supporto e utilità comuni per la manipolazione di immagini, conversioni di formato e operazioni matematiche ricorrenti.

- `metodi.py`  
  Factory per la creazione di istanze degli algoritmi di optical flow. Permette di selezionare e configurare dinamicamente l'algoritmo desiderato tramite nome o parametri.

- `opencvof.py`  
  Implementazione degli algoritmi di optical flow basati su OpenCV (es. Farneback, Lucas-Kanade). Consente la configurazione dettagliata dei parametri e l'integrazione con pipeline di elaborazione.

- `skiofilk.py`  
  Implementazione dell'algoritmo ILK (Iterative Lucas-Kanade) tramite scikit-image.

- `skioftvl1.py`  
  Implementazione dell'algoritmo TV-L1 (Total Variation L1) tramite scikit-image, robusto per immagini rumorose e variazioni di intensità.

- `skipccv.py`  
  Implementazione della Phase Cross-Correlation (PCC Vector) per la stima di spostamenti sub-pixel tra immagini. A differenza degli altri algoritmi che restituiscono mappe di spostamento in formato raster, `skipccv` restituirà un file georiferito vettoriale in cui ogni punto rappresenta il centro della finestra di ricerca, e sarà associato agli spostamenti nelle due direzioni principali (campi `RSHIFT` e `CSHIFT`) lo spostamento risultante (campo `L2`) e lo scarto quadratico medio normalizzato tra le finestre mobili analizzate (campo `NRMS`).

- `image_processing/`  
  Sotto-pacchetto con moduli per la normalizzazione, l'equalizzazione, la conversione e la manipolazione avanzata di immagini raster e array.

## Funzionalità principali
- Calcolo di optical flow tra immagini raster (OpenCV, scikit-image)
- Normalizzazione e trasformazione di bande e immagini
- Equalizzazione e conversione di immagini per l'analisi
- Interfacce astratte per la gestione di immagini multi-banda
- CLI per lanciare processi di tracking e normalizzazione
- Supporto per la conversione dei risultati in formati tabulari (DataFrame, GeoDataFrame)

## Esempio di utilizzo
```python
from sensetrack.ot.metodi import get_algorithm
from sensetrack.ot.interfaces import Image

ref = Image.from_file("ref.tif")
tar = Image.from_file("tar.tif")
alg = get_algorithm("OPENCVOF")
OT = alg.from_dict({"pyr_scale":0.5, "levels":4, "winsize":16})
result = OT(reference=ref.get_band("B1"), target=tar.get_band("B1"))
result.to_file("output.tif")
```

## CLI
Per lanciare l'analisi da terminale:
```powershell
python -m sensetrack.ot.cli --algname OPENCVOF --reference ref.tif --target tar.tif --output output.tif --winsize 16 --levels 4
```

## Algoritmi implementati

### OpenCV Optical Flow (Farneback, Lucas-Kanade)
- Implementato in `opencvof.py`.
- Algoritmi classici per optical flow densi e sparsi.
- Parametri configurabili: `pyr_scale`, `levels`, `winsize`, `iterations`, `poly_n`, `poly_sigma`, `flags`.
- Adatto a immagini con variazioni di intensità moderate e movimenti continui.

### Scikit-Image ILK (Iterative Lucas-Kanade)
- Implementato in `skiofilk.py`.
- Algoritmo iterativo per optical flow, robusto su immagini scientifiche e remote sensing.
- Parametri: `radius`, `num_warp`, `gaussian`, `prefilter`.
- Indicato per piccoli spostamenti e immagini con basso rumore.

### Scikit-Image TV-L1 (Total Variation L1)
- Implementato in `skioftvl1.py`.
- Algoritmo robusto per optical flow su immagini rumorose o con variazioni di intensità.
- Parametri: `attachment`, `tightness`, `num_warp`, `num_iter`, `tol`, `prefilter`.
- Ideale per dati SAR e immagini con discontinuità.

### Phase Cross-Correlation Vector (PCC Vector)
- Implementato in `skipccv.py`.
- Algoritmo per la stima di spostamenti sub-pixel tramite cross-correlazione di fase.
- Parametri: `winsize`, `step_size`, `phase_norm`, `upsmp_fac`.
- Utile per misurazioni di offset precisi tra immagini allineate.

## Note
- Tutti gli algoritmi sono configurabili tramite parametri Python o CLI.
- Il modulo è pensato per essere estendibile: è possibile aggiungere nuovi algoritmi implementando le interfacce previste.
- L'output può essere salvato in diversi formati, inclusi raster e DataFrame.

Per dettagli sui singoli algoritmi e parametri, consultare la documentazione inline nei file sorgente o l'help della CLI.
