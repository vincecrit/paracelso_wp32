import argparse

from ot.metodi import get_algorithm


def get_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description="Optical flow")
    # parametri comuni
    parser.add_argument("-ot", "--algname", help="Algoritmo da impiegare", type=str)
    parser.add_argument("-r", "--reference", help="Immagine di reference", type=str)
    parser.add_argument("-t", "--target", help="Immagine target", type=str)
    parser.add_argument("-o", "--output", help="Output", default='output.tif', type=str)
    parser.add_argument("-b", "--band", help="Banda da utilizzare, Se non indicata verranno utilizzate tutte le disponibili.", default=None, type=int)
    # parametri OpenCV
    parser.add_argument("--flow", help="Guess iniziale di flusso. Richiede l'argomento '--flags' impostato su 4", default=False)
    parser.add_argument("--levels", help="Numero di piramidi", default=4, type=int)
    parser.add_argument("--pyr_scale", help="Specifica il rapporto di scala da utilizzare tra un livello di piramidi e un altro.", default=0.5, type=float)
    parser.add_argument("--winsize", help="Dimensione finestra mobile", type=int, default=4)
    parser.add_argument("--iterations", help="Numero di iterazioni", default=10, type=int)
    parser.add_argument("--poly_n", help="Numero di pixel da usare per l'espansione pol", default=5, type=int)
    parser.add_argument("--poly_sigma", help="Deviazione standard della guassiana usata per smussare le derivate per l'espansione polinomiale.", default=1.1, type=float)
    parser.add_argument("--flags", help="Operazioni opzionali: '--flags 4' per utilizzare flusso iniziale (vedi --flow)\n'--flags 256' imposta un filtro gaussiano al posto di una box", default=None, type=int)
    # parametri scikit-image
    parser.add_argument("--radius", help="Dimensione finestra mobile", type=int, default=4)
    parser.add_argument("--num_warp", help="Numero di volte che la target viene sottoposta a warping", type=int, default=3)
    parser.add_argument("--gaussian", help="Utilizza un filtro gaussiano al posto di una box", action="store_true")
    parser.add_argument("--prefilter", help="Filtra l'immagine target prima di ogni warp", action="store_true")
    parser.add_argument("--attachment", help="Smussa il risultato finale quanto più è piccolo in valore", type=int, default=10)
    parser.add_argument("--tightness", help="Determina ", type=float, default=0.3)
    parser.add_argument("--num_iter", help="Numero fisso di iterazioni", type=int, default=10)
    parser.add_argument("--tol", help="Criterio di convergenza", type=float, default=1e-4)
    return parser


def main():
    args = get_parser().parse_args()
    alg = get_algorithm(args.algname)
    kwargs = vars(args)
    alg.run(**kwargs)


if __name__ == "__main__":
    main()
