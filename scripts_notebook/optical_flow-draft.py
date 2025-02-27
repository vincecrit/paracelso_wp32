
from ot import optical_flow
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import rasterio


def pixel_offset_to_displ(displacements: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray]:
    # displacement_meters = np.multiply(displacements, res)
    # x, y = displacement_meters.T

    px, py = displacements.T

    return px.T * dx, py.T * dy


def write_raster(fpath: str | Path, band: int, array: np.ndarray, metadata: dict) -> None:
    with rasterio.open(fp=str(fpath), mode='w', **metadata) as dst:
        dst.write_band(band, array)


def read_raster(filename: str | Path, band: int | None = None, **kwargs) -> tuple:
    with rasterio.open(filename, 'r', **kwargs) as rds:
        rds.read(band)


def get_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description="Optical flow")

    parser.add_argument("-r", "--reference", required=True,
                        help="Reference geotiff", type=str)

    parser.add_argument("-t", "--target", required=True,
                        help="Target geotiff", type=str)

    parser.add_argument("-o", "--output", required=False,
                        help="Output geotiff", default='output.tif', type=str)

    parser.add_argument("--flow", required=False, default=False)

    parser.add_argument("-nl", "--levels", required=False,
                        help="Numero di piramidi", default=4, type=int)

    parser.add_argument("-sl", "--pyr_scale",
                        required=False, default=0.5, type=float)

    parser.add_argument("-w", "--winsize", required=True,
                        help="Dimensione finestra mobile", type=int)

    parser.add_argument("-n", "--iterations", required=False,
                        help="Numero di iterazioni", default=10, type=int)

    parser.add_argument("-d", "--poly_n", required=False, default=5, type=int)

    parser.add_argument("-s", "--poly_sigma",
                        required=False, default=1.1, type=float)

    parser.add_argument("--flags", required=False, default=None)

    return parser


def main_old():

    parms = dict(
        flow=False,
        pyr_scale=0.5,
        levels=4,
        winsize=16,
        iterations=5,
        poly_n=5,
        poly_sigma=1.5,
        flags=None
    )

    print("Loading rasters")

    wd = "U:/critelli/tasks/corvara/gis/raster/"

    ref_path = wd+"SLOPE_2019_25cm.tif"
    target_path = wd+"SLOPE_2022_25cm.tif"

    ref = rasterio.open(ref_path).read()[0]
    target = rasterio.open(target_path).read()[0]

    metadata = rasterio.open(ref_path).meta
    metadata['dtype'] = 'float32'
    # # metadata['nodata'] = -9999

    mask = ~rasterio.open(ref_path).dataset_mask().astype(bool)

    print("Optical Flow")
    displacements = cv2.calcOpticalFlowFarneback(ref, target, **parms)
    dxx, dyy = pixel_offset_to_displ(displacements, metadata["transform"].a)

    res = np.linalg.norm([dxx, dyy], axis=0)
    res_ma = np.ma.masked_array(res, mask)

    write_raster(wd+'SLOPE_2019-2022_OF.tif', 1, res_ma, metadata)


def filename_to_date(string):
    yyyymmdd = re.findall(r"[0-9]+", string)

    return [int(p) for p in yyyymmdd]


def optical_flow_displacement(meta: dict, *args, **kwargs) -> tuple:

    pixel_offsets = cv2.calcOpticalFlowFarneback(*args, **kwargs)

    dxx, dyy = pixel_offset_to_displ(
        pixel_offsets,
        meta["transform"].a,
        -meta["transform"].e
    )

    return dxx, dyy, np.linalg.norm([dxx, dyy], axis=0)


def main():
    arg_parser = get_parser()

    args = arg_parser.parse_args()

    reference = rasterio.open(args.reference).read(1).astype(np.float32)
    target = rasterio.open(args.target).read(1).astype(np.float32)
    metadata_ref = rasterio.open(args.reference).meta

    pixel_offsets = cv2.calcOpticalFlowFarneback(
        reference, target,
        args.flow,
        args.pyr_scale,
        args.levels,
        args.winsize,
        args.iterations,
        args.poly_sigma,
        args.poly_n,
        args.flags
    )

    metadata_ref['dtype'] = 'float32'
    # # metadata['nodata'] = -9999

    dxx, dyy = pixel_offset_to_displ(
        pixel_offsets,
        metadata_ref["transform"].a, metadata_ref["transform"].a
    )

    res = np.linalg.norm([dxx, dyy], axis=0)

    mask = ~rasterio.open(args.reference).dataset_mask().astype(bool)
    res_ma = np.ma.masked_array(res, mask)

    write_raster(args.output, 1, res_ma, metadata_ref)


def main_new():
    arg_parser = get_parser()

    args = arg_parser.parse_args()

    reference = rasterio.open(args.reference).read(1).astype(np.float32)
    target = rasterio.open(args.target).read(1).astype(np.float32)
    meta = rasterio.open(args.reference).meta

    _, _, resdispl = optical_flow_displacement(
        meta,
        reference, target,
        args.flow,
        args.pyr_scale,
        args.levels,
        args.winsize,
        args.iterations,
        args.poly_n,
        args.poly_sigma,
        args.flags
    )

    mask = ~rasterio.open(args.reference).dataset_mask().astype(bool)

    meta['dtype'] = 'float32'
    resdispl_ma = np.ma.masked_array(resdispl, mask)

    write_raster(args.output, 1, resdispl_ma, meta)


# %%


def pixel_offset_to_displ(displacements: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray]:
    # displacement_meters = np.multiply(displacements, res)
    # x, y = displacement_meters.T

    px, py = displacements.T

    return px.T * dx, py.T * dy


args = {
    "reference": "data/HSD_2019.tif",
    "target": "data/HSD_2021.tif",
    "output": "data/output2.tif",
    "flow": False,
    "pyr_scale": 0.5,
    "levels": 4,
    "winsize": 5,
    "iterations": 5,
    "poly_n": 5,
    "poly_sigma": 1.1,
    "flags": None
}

of = optical_flow.OpticalFlowFarneback(
    flow=args["flow"],
    pyr_scale=args["pyr_scale"],
    levels=args["levels"],
    winsize=args["winsize"],
    iterations=args["iterations"],
    poly_n=args["poly_n"],
    poly_sigma=args["poly_sigma"],
    flags=args["flags"]
)

reference, _ = read_raster(args["reference"], 1)
target, meta = read_raster(args["target"], 1)


# Path("parms.json").write_text(of.toJSON())

# DX = DY = meta['transform'].a

# sx, sy = pixel_offset_to_displ(of(reference, target), DX, DY)
# r = np.linalg.norm([sx, sy], axis=0)
# r_ma = np.ma.masked_array(r, reference.mask)

# print(reference.shape)
# print(meta)

# images_io.write_raster(args["output"], r_ma, meta)



def main2():

    args = get_parser().parse_args()

    alg = get_algorithm(args.algname)

    reference = rasterio.open(args.reference).read(1).astype(np.float32)
    target = rasterio.open(args.target).read(1).astype(np.float32)
    meta = rasterio.open(args.reference).meta

    otalg = alg.from_dict(vars(args))

    try:
        otalg.check()

    except NotImplementedError:
        pass

    resdispl = otalg(meta, reference, target)

    mask = ~rasterio.open(args.reference).dataset_mask().astype(bool)

    meta['dtype'] = 'float32'
    resdispl_ma = np.ma.masked_array(resdispl, mask)

    write_raster(args.output, 1, resdispl_ma, meta)


def main(algname: str, i1: str | Path, i2: str | Path, parms: dict | Path | str):

    if isinstance(parms, dict):
        otalg = get_algorithm(algname).from_dict(parms)

    elif isinstance(parms, (Path, str)):

        if Path(parms).suffix.endswith('json'):
            otalg = get_algorithm(algname).from_JSON(parms)

        elif Path(parms).suffix.endswith('yaml'):
            print(1)
            otalg = get_algorithm(algname).from_YAML(parms)

        else:
            otalg = None

    try:
        otalg.check()

    except NotImplementedError:
        pass

    # i1, i2 = read_raster(i1, 1), read_raster(i2, 1)
    displacements = otalg(i1, i2)

    # write_raster(displacements)

# %%


if __name__ == "__main__":
    main_new()
