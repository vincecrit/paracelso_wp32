# sensetrack.s1 - Sentinel-1 SAR Preprocessing Submodule

## Overview
Il sottopacchetto `sensetrack.sentinel` fornisce strumenti e classi per il preprocessamento di dati SAR provenienti dal satellite Sentinel-1. Consente la gestione automatizzata di workflow di elaborazione, la lettura e manipolazione dei manifest, e l'integrazione con pipeline di analisi più ampie.

## Struttura del modulo
- `preprocessing.py`
  - Classe principale: `S1Preprocessor`
  - Gestisce l'intero workflow di preprocessamento per dati Sentinel-1 SLC e GRD.
  - Permette la configurazione di parametri, la selezione di aree di interesse (AOI) e l'integrazione con SNAP-GPT.
  - Supporta l'esecuzione batch su più file SAR.

- `manifest_file.py`
  - Funzioni e classi per la lettura, il parsing e la manipolazione dei file manifest XML di Sentinel-1.
  - Estrae metadati, parametri di acquisizione, geometrie e informazioni di orbita utili per il preprocessamento e l'analisi.

## Funzionalità principali
- **Preprocessamento SAR Sentinel-1**: automatizza la catena di elaborazione (calibrazione, ortorettifica, filtraggio, subset AOI, ecc.)
- **Gestione manifest**: parsing avanzato dei file manifest XML per estrarre metadati e parametri di scena.
- **Configurazione flessibile**: parametri e workflow personalizzabili tramite file di configurazione e classi Python.
- **Integrazione con SNAP-GPT**: utilizzo di grafi XML per lanciare processi SNAP direttamente da Python.
- **Supporto batch**: elaborazione di più scene in sequenza automatica.

## Esempio di utilizzo
```python
from sensetrack.sentinel.preprocessing import S1Preprocessor

# Crea un preprocessor per una specifica AOI e processo
preproc = S1Preprocessor(aoi="CALITA", process="S1_SLC_DEFAULT")
preproc.run("path/to/sarfile.zip")
```

## Note
- Il modulo richiede che SNAP sia installato e accessibile dal sistema.
- I workflow possono essere estesi per includere ulteriori step di elaborazione (ad es. speckle filtering, coregistrazione, ecc.).
- Il parsing dei manifest consente di automatizzare la selezione di parametri e la verifica di qualità dei dati.

Per dettagli sui parametri e sulle opzioni disponibili, consultare la documentazione inline nei file sorgente e i commenti nei file di configurazione YAML.
