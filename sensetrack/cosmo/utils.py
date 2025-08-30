"""
utils.py
This module provides utility classes and functions for
string manipulation within the sensetrack.cosmo package.
"""
import math
import statistics
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import shapely

from sensetrack.cosmo import lib


def show_dataset_attr(h5_file, path_to_dataset):
    with h5py.File(h5_file, "r") as f:
        dset = f[path_to_dataset]
        attrs = dset.attrs

        for key in attrs:
            print(f"{key}: {attrs[key]}")


def visit_h5(h5_file):

    with h5py.File(h5_file, "r") as f:
        def show_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}")
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")

        f.visititems(show_structure)


def footprint_polygon(*args) -> shapely.Geometry:
    y, x, _ = np.array(args).T
    return shapely.Polygon(np.c_[x, y])


def csg_footprint(h5_file):
    with h5py.File(h5_file, "r") as f:
        attrs = f['/S01/IMG'].attrs
        tl = attrs['Top Left Geodetic Coordinates']
        bl = attrs['Bottom Left Geodetic Coordinates']
        br = attrs['Bottom Right Geodetic Coordinates']
        tr = attrs['Top Right Geodetic Coordinates']

    return tl, bl, br, tr


def csk_footprint(h5_file):
    with h5py.File(h5_file, "r") as f:
        attrs = f['/S01/SBI'].attrs
        tl = attrs['Top Left Geodetic Coordinates']
        bl = attrs['Bottom Left Geodetic Coordinates']
        br = attrs['Bottom Right Geodetic Coordinates']
        tr = attrs['Top Right Geodetic Coordinates']

    return tl, bl, br, tr


def csk_shape(h5_file):
    with h5py.File(h5_file, "r") as f:
        SBI = f['/S01/SBI']
        if isinstance(SBI, h5py.Dataset):
            return SBI.shape
        else:
            raise TypeError("'/S01/SBI' is not a dataset")


def csg_shape(h5_file):
    with h5py.File(h5_file, "r") as f:
        IMG = f['/S01/IMG']
        if isinstance(IMG, h5py.Dataset):
            return IMG.shape
        else:
            raise TypeError("'/S01/IMG' is not a dataset")
        

def csk_mean_incidence_angle_rad(h5_file) -> float | None:
    with h5py.File(h5_file, "r") as f:
        dset = f['/S01/SBI']
        attrs = dset.attrs
        far_iangle = attrs.get('Far Incidence Angle', None)
        near_iangle = attrs.get('Near Incidence Angle', None)

        if far_iangle and near_iangle:
            return statistics.mean([far_iangle, near_iangle]) * (math.pi/180.)
        else:
            return


def csg_mean_incidence_angle_rad(h5_file) -> float | None:
    with h5py.File(h5_file, "r") as f:
        dset = f['/']
        attrs = dset.attrs
        far_iangle = attrs.get('Far Incidence Angle', None)
        near_iangle = attrs.get('Near Incidence Angle', None)

        if far_iangle and near_iangle:
            return statistics.mean([far_iangle, near_iangle]) * (math.pi/180.)
        else:
            return


def _get_attributes(h5: h5py.File) -> dict:

    attributes = dict()
    for attr in list(h5.attrs):
        attributes[attr] = h5.attrs[attr]

    return attributes


def _get_group_names(h5) -> list:
    return list(h5.keys())


def _get_attrs_names(h5) -> list:
    return list(h5.attrs)


def _exploreh5(h5):
    pf = '   '
    print(type(h5))
    attrtype = [f"{pf}{a}: {type(h5.attrs[a])}" for a in _get_attrs_names(h5)]
    print('\n'.join(attrtype))

    try:
        for group in _get_group_names(h5):
            print(group)
            _exploreh5(h5[f'{group}'])
    except AttributeError:
        pass


def qlk_to_images(h5_files: list, wd: str | Path = Path.cwd(),
                  format: str = 'jpeg') -> None:
    '''
    Esporta una lista di file CSK (formato H5) nel formato
    indicato dall'argomento `format` (default = `jpeg`)
    '''
    for csk_file in h5_files:
        if not Path(csk_file).is_file():
            raise FileNotFoundError(f"File not found: {csk_file}")

    wd = Path(wd)
    if not wd.is_dir():
        wd.mkdir(parents=True)

    for csk_file in h5_files:
        jpeg = wd/f"{csk_file.stem}.{format}"
        print(f"Esporto: {jpeg}")
        lib.CSKFile(csk_file).qlk_to_image.save(jpeg)


def footprints_to_geopandas(csk_files: list[str]) -> gpd.GeoDataFrame:

    geoms = [lib.CSKFile(file).footprint_polygon for file in csk_files]

    return gpd.GeoDataFrame(dict(), crs='EPSG:4326', geometry=geoms)
