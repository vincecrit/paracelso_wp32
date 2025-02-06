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
NORMALIZE = '''Normalizza le immagini rispetto ai valori minimi e massimi'''
POWERNORM = '''Normalizza le immagini con potenza. Indica il parametro gamma'''
FLOW = '''Guess iniziale di flusso. Richiede l'argomento '--flags' impostato
su 4'''
LEVELS = '''Numero di piramidi'''
PYR_SCALE = '''Specifica il rapporto di scala da utilizzare tra un livello di
 piramidi e un altro.'''
WINSIZE = '''Dimensione finestra mobile'''
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