from datetime import datetime
from pathlib import Path

import geopandas as gpd
import h5py

import cosmo


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
    wd = Path(wd)
    if not wd.is_dir():
        wd.mkdir(parents=True)

    for csk_file in h5_files:
        jpeg = wd/f"{csk_file.stem}.{format}"
        print(f"Esporto: {jpeg}")
        cosmo.CSKFile(csk_file).qlk_to_image.save(jpeg)


def footprints_to_geopandas(csk_files: list[str]) -> gpd.GeoDataFrame:

    geoms = [cosmo.CSKFile(file).footprint_polygon for file in csk_files]

    return gpd.GeoDataFrame(dict(), crs='EPSG:4326', geometry=geoms)
