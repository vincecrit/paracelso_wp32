# sensetrack.snap_gpt - SNAP-GPT Submodule

## Overview
Il sottopacchetto `sensetrack.snap_gpt` fornisce strumenti e workflow per il preprocessamento di dati SAR (Synthetic Aperture Radar) tramite l'integrazione con SNAP-GPT (Graph Processing Tool) di ESA. Permette la gestione automatizzata di grafi di elaborazione, la definizione di aree di interesse (AOI), la configurazione di parametri e l'esecuzione batch di processi SAR.

## Struttura del modulo
- `lib.py`
  - Funzioni e classi per la gestione dei workflow SNAP-GPT, inclusa la creazione, modifica ed esecuzione di grafi di elaborazione.
  - Gestione delle chiamate a SNAP-GPT da Python, parsing dei log e gestione degli errori.
  - Utility per la generazione di comandi, la gestione dei file temporanei e la verifica dei risultati.

- `sarprep_config.yaml`
  - File di configurazione YAML per la definizione dei parametri di preprocessamento, delle cartelle di output, dei percorsi ai grafi SNAP e delle AOI.
  - Permette la personalizzazione dei workflow senza modificare il codice Python.

- `aoi.gpkg`
  - File GeoPackage contenente le aree di interesse (AOI) utilizzabili nei workflow SNAP-GPT.

## Funzionalità principali
- **Gestione workflow SNAP-GPT**: automatizza la creazione e l'esecuzione di grafi SNAP per preprocessamento SAR (ad es. Sentinel-1, COSMO-SkyMed).
- **Configurazione flessibile**: parametri, percorsi e AOI sono gestiti tramite file YAML e GeoPackage.
- **Supporto multi-sensore**: workflow predefiniti per diversi sensori (S1, COSMO, S2) tramite file XML di grafo.
- **Batch processing**: esecuzione di processi su più file SAR in modo automatico e ripetibile.
- **Gestione AOI**: selezione e applicazione di aree di interesse per il subset delle immagini.
- **Logging e gestione errori**: parsing dei log SNAP-GPT, gestione delle eccezioni e reportistica.

## Esempio di utilizzo
```python
from sensetrack.snap_gpt.lib import SARPreprocessing, AOI

# Seleziona area di interesse e processo
subset = AOI.CALITA
process = SARPreprocessing.S1_SLC_DEFAULT
sarfile = "path/to/sarfile.zip"

# Crea e lancia il preprocessamento
preproc = SARPreprocessing(subset, process)
preproc.run(sarfile)
```

## File di configurazione
- `sarprep_config.yaml` contiene parametri come:
  - OUTFOLDER: cartella di output
  - GRAPHS_WD: percorso ai grafi SNAP
  - AOI_GPKG: percorso al file delle AOI
- Esempio di struttura YAML:
  ```yaml
  OUTFOLDER: ./output
  GRAPHS_WD: ./graphs
  AOI_GPKG: ./aoi.gpkg
  ```

## Note
- I grafi SNAP (file XML) sono personalizzabili e possono essere estesi per nuovi workflow.
- Il modulo è pensato per essere integrato in pipeline più ampie di elaborazione dati SAR.
- Per l'uso di SNAP-GPT è necessario che SNAP sia installato e accessibile dal sistema.

Per dettagli sui parametri e sulle opzioni disponibili, consultare la documentazione inline nei file sorgente e i commenti nei file di configurazione YAML.
