'''
Help strings for parser function
'''

ALGNAME = '''Algoritmo da impiegare'''
REFERENCE = '''Immagine di reference'''
TARGET = '''Immagine target'''
OUTPUT = '''Output file'''
NODATA = '''Valore di nodata'''
BAND = '''Banda da utilizzare, Se non indicata verranno utilizzate tutte le
disponibili.'''
RGB2GRAY = '''Converte le immagini RGB in scala di grigi (banda singola)'''
LOGNORM = '''Normalizza i logaritmi delle immagini. Ha la precedenza su
'--normalize' e '--powernorm'''
MINMAX = '''Normalizza le singole bande nel range 0 - 1 rispetto ai valori minimo e massimo'''
CLAHE = '''Normalizza le singole bande attraverso un algoritmo CLAHE (Contrast Limited Adaptive Histogram Equalization)'''
NORMALIZE = '''Normalizza le immagini rispetto ai valori minimi e massimi'''
ZSCORENORM = '''Normalizza le immagini con potenza. Indica il parametro gamma'''
FLOW = '''Guess iniziale di flusso. Richiede l'argomento '--flags' impostato
su 4'''
LEVELS = '''Numero di piramidi'''
PYR_SCALE = '''Specifica il rapporto di scala da utilizzare tra un livello di
 piramidi e un altro.'''
WINSIZE = '''Dimensione finestra mobile'''
STEPSIZE = '''intervallo di campionamento. L'immagine verrà scomposta in tante
 finestre mobili centrate su punti dell'immagine distanziati di `step_size` pixels.'''
ITERATIONS = '''Numero di iterazioni'''
POLY_N = '''Numero di pixel da usare per l'espansione pol'''
POLY_SIGMA = '''Deviazione standard della guassiana usata per smussare le
derivate per l'espansione polinomiale.'''
FLAGS = '''Operazioni opzionali: '--flags 4' per utilizzare flusso iniziale
(vedi --flow)\n'--flags 256' imposta un filtro gaussiano al posto di una box'''
RADIUS = '''Dimensione finestra mobile'''
NUMWARP = '''Numero di iterazioni per il warp'''
GAUSSIAN = '''Applica un filtro gaussiano all'immagine di output'''
PREFILTER = '''Applica un filtro di pre-filtraggio all'immagine di output'''
ATTACHMENT = '''Smussa il risultato finale quanto è piu' piccolo in valore'''
TIGHTNESS = '''Determina il valore di tightness'''
NUMITER = '''Numero fisso di iterazioni'''
TOL = '''Tolleranza per la convergenza'''
PHASENORM = '''Tipo di normalizzazione nella cross-correlazione. Se `True` applica
una normalizzazione FFT alle finestre mobili'''
UPSAMPLE_FACTOR = '''Utile per identificare spostamenti a scala di sub-pixel.
Influenza molto il carico di calcolo necessario.'''