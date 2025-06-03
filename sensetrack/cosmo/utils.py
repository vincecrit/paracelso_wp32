"""
utils.py
This module provides utility classes and functions for string manipulation within the sensetrack.cosmo package.
"""
import math
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import shapely

from sensetrack.cosmo import lib


class StrChopper:
    def __init__(self, s: str) -> None:
        self._s = s

    @property
    def s(self): return self._s

    @s.setter
    def s(self, value: str): self._s = value

    def chop(self, n):
        l = list(self.s)

        chunck = "".join([l.pop(0) for _ in range(n)])

        self.s = "".join(l)

        return chunck


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
        SBI = f['/S01/SBI'][()]

    return SBI.shape


def csg_shape(h5_file):
    with h5py.File(h5_file, "r") as f:
        IMG = f['/S01/IMG'][()]

    return IMG.shape


def csk_mean_incidence_angle_rad(h5_file):
    with h5py.File(h5_file, "r") as f:
        dset = f['/S01/SBI']
        attrs = dset.attrs
        far_iangle = float(attrs['Far Incidence Angle'])
        near_iangle = float(attrs['Near Incidence Angle'])
        return (sum([far_iangle, near_iangle])/2 * math.pi) / 180.


def csg_mean_incidence_angle_rad(h5_file):
    with h5py.File(h5_file, "r") as f:
        dset = f['/']
        attrs = dset.attrs
        far_iangle = float(attrs['Far Incidence Angle'])
        near_iangle = float(attrs['Near Incidence Angle'])

        return (sum([far_iangle, near_iangle])/2 * math.pi) / 180.


def str2dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d%H%M%S")


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


def batch_to_image(h5_files: list, wd: str | Path = Path.cwd(),
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
        lib.CSKProduct(csk_file).qlk_to_image.save(jpeg)


def footprints_to_geopandas(csk_files: list[str]) -> gpd.GeoDataFrame:

    geoms = [lib.CSKProduct(file).footprint_polygon for file in csk_files]

    return gpd.GeoDataFrame(dict(), crs='EPSG:4326', geometry=geoms)
