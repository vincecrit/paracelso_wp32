import argparse

from ot.metodi import get_algorithm


def get_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description="Optical flow")
    # parametri comuni
    parser.add_argument("-ot", "--algname", required=False, help="Algoritmo da impiegare", type=str)
    parser.add_argument("-r", "--reference", required=False, help="Immagine di reference", type=str)
    parser.add_argument("-t", "--target", required=False, help="Immagine target", type=str)
    parser.add_argument("-o", "--output", required=False, help="Output", default='output.tif', type=str)
    # parametri OpenCV
    parser.add_argument("--flow", required=False, default=False)
    parser.add_argument("--levels", required=False, help="Numero di piramidi", default=4, type=int)
    parser.add_argument("--pyr_scale", required=False, default=0.5, type=float)
    parser.add_argument("--winsize", required=False, help="Dimensione finestra mobile", type=int, default=4)
    parser.add_argument("--iterations", required=False, help="Numero di iterazioni", default=10, type=int)
    parser.add_argument("--poly_n", required=False, default=5, type=int)
    parser.add_argument("--poly_sigma", required=False, default=1.1, type=float)
    parser.add_argument("--flags", required=False, default=None, type=int)
    # parametri scikit-image
    parser.add_argument("--radius", required=False, help="Dimensione finestra mobile", type=int, default=4)
    parser.add_argument("--attachment", required=False, type=int, default=10)
    parser.add_argument("--tightness", required=False, type=float, default=0.3)
    parser.add_argument("--num_warp", required=False, type=int, default=3)
    parser.add_argument("--num_iter", required=False, type=int, default=10)
    parser.add_argument("--tol", required=False, type=float, default=1e-4)
    parser.add_argument("--gaussian", required=False, action="store_true")
    parser.add_argument("--prefilter", required=False, action="store_true")
    return parser


def main():
    args = get_parser().parse_args()
    alg = get_algorithm(args.algname)
    alg.run(**vars(args))


if __name__ == "__main__":
    main()
